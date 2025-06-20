import torch
import numpy as np
from scipy.special import psi, gamma
from ele2364 import Memory
from einops import rearrange, unpack

class Memory_tensors(Memory):
    def __init__(self,ensemble_size,**args):
        super().__init__(**args)
        self.ensemble_size=ensemble_size

    def sample(self,size):
        """Get random minibatch from memory.
        
        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        idx = np.random.randint(0, self.n, (self.ensemble_size,size))
        #self.s_sample=rearrange(self.s[idx],"d es bs -> es bs d")
        #self.a_sample=rearrange(self.a[idx],"d es bs -> es bs d")
        #rearrange(self.r[idx],"d es bs -> es bs d")
        #self.sp_sample=rearrange(self.sp[idx],"d es bs -> es bs d")
        #rearrange(self.terminal[idx],"d es bs -> es bs d")

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.terminal[idx] # [d es bs]



class InformationGain(object):
    def __init__(self, model, scale=1.0):
        self.model = model
        self.scale = scale

    def __call__(self, delta_means, delta_vars):
        """
        delta_means   (plan_horizon, ensemble_size, n_candidates, n_dim)
        delta_vars    (plan_horizon, ensemble_size, n_candidates, n_dim)
        """

        plan_horizon = delta_means.size(0)
        n_candidates = delta_means.size(2)

        delta_means = self.model.normalizer.renormalize_state_delta_means(delta_means)
        delta_vars = self.model.normalizer.renormalize_state_delta_vars(delta_vars)
        delta_states = self.model.sample(delta_means, delta_vars)
        info_gains = (
            torch.zeros(plan_horizon, n_candidates).float().to(delta_means.device)
        )

        for t in range(plan_horizon):
            ent_avg = self.entropy_of_average(delta_states[t])
            avg_ent = self.average_of_entropy(delta_vars[t])
            info_gains[t, :] = ent_avg - avg_ent

        info_gains = info_gains * self.scale
        return info_gains.sum(dim=0)

    def entropy_of_average(self, samples):
        """
        samples (ensemble_size, n_candidates, n_dim) 
        """
        samples = samples.permute(1, 0, 2)
        n_samples = samples.size(1)
        dims = samples.size(2)
        k = 3

        distances_yy = self.batched_cdist_l2(samples, samples)
        y, _ = torch.sort(distances_yy, dim=1)
        v = self.volume_of_the_unit_ball(dims)
        h = (
            np.log(n_samples - 1)
            - psi(k)
            + np.log(v)
            + dims * torch.sum(torch.log(y[:, k - 1]), dim=1) / n_samples
            + 0.5
        )
        return h

    def batched_cdist_l2(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = (
            torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2)
            .add_(x1_norm)
            .clamp_min_(1e-30)
            .sqrt_()
        )
        return res

    def volume_of_the_unit_ball(self, dim):
        return np.pi ** (dim / 2) / gamma(dim / 2 + 1)

    def average_of_entropy(self, delta_vars):
        return torch.mean(self.gaussian_diagonal_entropy(delta_vars), dim=0)

    def gaussian_diagonal_entropy(self, delta_vars):
        min_variance = 1e-8
        return 0.5 * torch.sum(
            torch.log(2 * np.pi * np.e * torch.clamp(delta_vars, min=min_variance)),
            dim=len(delta_vars.size()) - 1,
        )
    




class Normalizer(object):
    def __init__(self):
        self.state_mean = None
        self.state_sk = None
        self.state_stdev = None
        self.action_mean = None
        self.action_sk = None
        self.action_stdev = None
        self.state_delta_mean = None
        self.state_delta_sk = None
        self.state_delta_stdev = None
        self.count = 0

    @staticmethod
    def update_mean(mu_old, addendum, n):
        mu_new = mu_old + (addendum - mu_old) / n
        return mu_new

    @staticmethod
    def update_sk(sk_old, mu_old, mu_new, addendum):
        sk_new = sk_old + (addendum - mu_old) * (addendum - mu_new)
        return sk_new

    def update(self, state, action, state_delta):
        self.count += 1

        if self.count == 1:
            self.state_mean = state.copy()
            self.state_sk = np.zeros_like(state)
            self.state_stdev = np.zeros_like(state)
            self.action_mean = action.copy()
            self.action_sk = np.zeros_like(action)
            self.action_stdev = np.zeros_like(action)
            self.state_delta_mean = state_delta.copy()
            self.state_delta_sk = np.zeros_like(state_delta)
            self.state_delta_stdev = np.zeros_like(state_delta)
            return

        state_mean_old = self.state_mean.copy()
        action_mean_old = self.action_mean.copy()
        state_delta_mean_old = self.state_delta_mean.copy()

        self.state_mean = self.update_mean(self.state_mean, state, self.count)
        self.action_mean = self.update_mean(self.action_mean, action, self.count)
        self.state_delta_mean = self.update_mean(
            self.state_delta_mean, state_delta, self.count
        )

        self.state_sk = self.update_sk(
            self.state_sk, state_mean_old, self.state_mean, state
        )
        self.action_sk = self.update_sk(
            self.action_sk, action_mean_old, self.action_mean, action
        )
        self.state_delta_sk = self.update_sk(
            self.state_delta_sk,
            state_delta_mean_old,
            self.state_delta_mean,
            state_delta,
        )

        self.state_stdev = np.sqrt(self.state_sk / self.count)
        self.action_stdev = np.sqrt(self.action_sk / self.count)
        self.state_delta_stdev = np.sqrt(self.state_delta_sk / self.count)

    @staticmethod
    def setup_vars(x, mean, stdev):
        mean, stdev = mean.copy(), stdev.copy()
        mean = torch.from_numpy(mean).float().to(x.device)
        stdev = torch.from_numpy(stdev).float().to(x.device)
        return mean, stdev

    def _normalize(self, x, mean, stdev):
        mean, stdev = self.setup_vars(x, mean, stdev)
        n = x - mean
        n = n / torch.clamp(stdev, min=1e-8)
        return n

    def normalize_states(self, states):
        return self._normalize(states, self.state_mean, self.state_stdev)

    def normalize_actions(self, actions):
        return self._normalize(actions, self.action_mean, self.action_stdev)

    def normalize_state_deltas(self, state_deltas):
        return self._normalize(
            state_deltas, self.state_delta_mean, self.state_delta_stdev
        )

    def denormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.setup_vars(
            state_deltas_means, self.state_delta_mean, self.state_delta_stdev
        )
        return state_deltas_means * stdev + mean

    def denormalize_state_delta_vars(self, state_delta_vars):
        _, stdev = self.setup_vars(
            state_delta_vars, self.state_delta_mean, self.state_delta_stdev
        )
        return state_delta_vars * (stdev ** 2)

    def renormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.setup_vars(
            state_deltas_means, self.state_delta_mean, self.state_delta_stdev
        )
        return (state_deltas_means - mean) / torch.clamp(stdev, min=1e-8)

    def renormalize_state_delta_vars(self, state_delta_vars):
        _, stdev = self.setup_vars(
            state_delta_vars, self.state_delta_mean, self.state_delta_stdev
        )
        return state_delta_vars / (torch.clamp(stdev, min=1e-8) ** 2)