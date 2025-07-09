from ele2364.networks import Pi, V, ttf, tti
from torch import nn
from .networks import *
from .utils import InformationGain
import numpy as np
from einops import rearrange


class Planner(nn.Module):
    def __init__(
        self,
        ensemble,
        reward_model,
        action_size,
        ensemble_size,
        plan_horizon,
        optimisation_iters,
        n_candidates,
        top_candidates,
        #buffer,
        batch_size,
        learning_rate,
        epsilon=0.99,
        use_reward=True,
        use_exploration=True,
        use_mean=False,
        expl_scale=1.0,
        reward_scale=1.0,
        strategy="information",
        device="cpu",
    ):
        super().__init__()
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.action_size = action_size
        self.ensemble_size = ensemble_size

        self.plan_horizon = plan_horizon
        self.optimisation_iters = optimisation_iters
        self.n_candidates = n_candidates
        self.top_candidates = top_candidates

        self.use_reward = use_reward
        self.use_exploration = use_exploration
        self.use_mean = use_mean
        self.expl_scale = expl_scale
        self.reward_scale = reward_scale
        self.device = device

        if strategy == "information":
            self.measure = InformationGain(self.ensemble, scale=expl_scale)
        #elif strategy == "variance":
        #    self.measure = Variance(self.ensemble, scale=expl_scale)
        #elif strategy == "random":
        #    self.measure = Random(self.ensemble, scale=expl_scale)
        #elif strategy == "none":
        #    self.use_exploration = False

        self.trial_rewards = []
        self.trial_bonuses = []
        self.to(device)

        #self.buffer=buffer
        self.batch_size=batch_size

        self.params = list(ensemble.parameters()) + list(reward_model.parameters())
        self.optim = torch.optim.Adam(self.params, lr=learning_rate, eps=epsilon)

    def forward(self, state):

        state = torch.from_numpy(state).float().to(self.device)
        state_size = state.size(0)

        action_mean = torch.zeros(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        action_std_dev = torch.ones(self.plan_horizon, 1, self.action_size).to(
            self.device
        )
        for _ in range(self.optimisation_iters):
            actions = action_mean + action_std_dev * torch.randn(  # [T, n_can, act_size]
                self.plan_horizon,
                self.n_candidates,
                self.action_size,
                device=self.device,
            )
            states, delta_vars, delta_means = self.perform_rollout(state, actions)

            returns = torch.zeros(self.n_candidates).float().to(self.device)
            if self.use_exploration:
                expl_bonus = self.measure(delta_means, delta_vars) * self.expl_scale # [T B n_can states]
                returns += expl_bonus
                #self.trial_bonuses.append(expl_bonus)

            if self.use_reward:
                _states = states.view(-1, state_size)
                _actions = actions.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1)
                _actions = _actions.view(-1, self.action_size)
                rewards = self.reward_model(_states, _actions)
                rewards = rewards * self.reward_scale
                rewards = rewards.view(
                    self.plan_horizon, self.ensemble_size, self.n_candidates
                )
                rewards = rewards.mean(dim=1).sum(dim=0)
                returns += rewards
                #self.trial_rewards.append(rewards)

            action_mean, action_std_dev = self._fit_gaussian(actions, returns)

        return action_mean[0].squeeze(dim=0).numpy()

    def perform_rollout(self, current_state, actions):
        T = self.plan_horizon + 1
        states = [torch.empty(0)] * T
        delta_means = [torch.empty(0)] * T
        delta_vars = [torch.empty(0)] * T

        current_state = current_state.unsqueeze(dim=0).unsqueeze(dim=0) # [1 1 states]
        current_state = current_state.repeat(self.ensemble_size, self.n_candidates, 1) # [B n_can, states]
        states[0] = current_state

        actions = actions.unsqueeze(0) # [1 T, n_can, act_size]
        actions = actions.repeat(self.ensemble_size, 1, 1, 1).permute(1, 0, 2, 3) # [T B n_can act_size]

        for t in range(self.plan_horizon):
            delta_mean, delta_var = self.ensemble(states[t], actions[t]) # Input: [B n_can states], [B n_can act_size] out: [B n_can state]
            if self.use_mean:
                states[t + 1] = states[t] + delta_mean
            else:
                states[t + 1] = states[t] + self.ensemble.sample(delta_mean, delta_var)
            delta_means[t + 1] = delta_mean
            delta_vars[t + 1] = delta_var

        states = torch.stack(states[1:], dim=0) # [T B n_can states]
        delta_vars = torch.stack(delta_vars[1:], dim=0) # [T B n_can states]
        delta_means = torch.stack(delta_means[1:], dim=0) # [T B n_can states]
        return states, delta_vars, delta_means

    def _fit_gaussian(self, actions, returns):
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
        _, topk = returns.topk(self.top_candidates, dim=0, largest=True, sorted=False)
        best_actions = actions[:, topk.view(-1)].reshape(
            self.plan_horizon, self.top_candidates, self.action_size
        )
        action_mean, action_std_dev = (
            best_actions.mean(dim=1, keepdim=True),
            best_actions.std(dim=1, unbiased=False, keepdim=True),
        )
        return action_mean, action_std_dev

    def return_stats(self):
        if self.use_reward:
            reward_stats = self._create_stats(self.trial_rewards)
        else:
            reward_stats = {}
        if self.use_exploration:
            info_stats = self._create_stats(self.trial_bonuses)
        else:
            info_stats = {}
        self.trial_rewards = []
        self.trial_bonuses = []
        return reward_stats, info_stats

    def _create_stats(self, arr):
        tensor = torch.stack(arr)
        tensor = tensor.view(-1)
        return {
            "max": tensor.max().item(),
            "min": tensor.min().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
        }
    
    def train(self,n_train_epochs,memory):
        e_losses = []
        r_losses = []
        n_batches = []
        for epoch in range(1, n_train_epochs + 1):
            e_losses.append([])
            r_losses.append([])
            n_batches.append(0)

            
            #for (states, actions, rewards, deltas) in self.buffer(self.batch_size):
            for i in range(len(memory)//self.batch_size):
                #(s, actions, rewards, sp, terminal)=np.array([list(memory.sample(self.batch_size)) for _ in self.ensemble_size]) # [[bs d],[bs d],[bs d]] [es 5 bs d]
                s, actions, rewards, sp, terminal=memory.sample(self.batch_size)
                
                deltas=sp - s
                self.ensemble.train()
                self.reward_model.train()

                self.optim.zero_grad()
                e_loss = self.ensemble.loss(ttf(s).to(self.device), ttf(actions).to(self.device), ttf(deltas).to(self.device))
                r_loss = self.reward_model.loss(ttf(s).to(self.device), ttf(actions).to(self.device), ttf(rewards).to(self.device))
                (e_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.params, 1.0, norm_type=2
                )
                self.optim.step()

                e_losses[epoch - 1].append(e_loss.item())
                r_losses[epoch - 1].append(r_loss.item())
                n_batches[epoch - 1] += 1

            #if self.logger is not None and epoch % 20 == 0:
            #    avg_e_loss = self._get_avg_loss(e_losses, n_batches, epoch)
            #    avg_r_loss = self._get_avg_loss(r_losses, n_batches, epoch)
            #    message = "> Train epoch {} [ensemble {:.2f} | reward {:.2f}]"
            #    self.logger.log(message.format(epoch, avg_e_loss, avg_r_loss))

        return (
            self._get_avg_loss(e_losses, n_batches, epoch),
            self._get_avg_loss(r_losses, n_batches, epoch),
        )

    def reset_models(self):
        self.ensemble.reset_parameters()
        self.reward_model.reset_parameters()
        self.params = list(self.ensemble.parameters()) + list(
            self.reward_model.parameters()
        )
        self.optim = torch.optim.Adam(
            self.params, lr=self.learning_rate, eps=self.epsilon
        )

    def _get_avg_loss(self, losses, n_batches, epoch):
        epoch_loss = [sum(loss) / n_batch for loss, n_batch in zip(losses, n_batches)]
        return sum(epoch_loss) / epoch