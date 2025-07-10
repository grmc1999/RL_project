

from random import choice
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ele2364.networks import Pi, V
from ele2364 import Memory
import time
from einops import rearrange

EPOCHS=80
BATCH_SIZE=2000
GAMMA=0.99
LAMBDA=0.97


class GRPO(object):
    def __init__(self,actor,environment,mem_args):
        self.env=environment
        self.actor=actor
        #self.critic=critic
        self.mem_args=mem_args
        self.DATA=Memory(*mem_args)
        #self.model_learning_epochs=model_learning_epochs

        self.transform_data=(lambda mem_mod,last_index,ep_size:rearrange(mem_mod[:last_index],"(ep steps) 1 -> ep steps 1",steps=int(ep_size)))
    
    def train(self,episodes):

        episodes=tqdm(range(episodes))
        steps=tqdm(range(1000))
        t=0
        total_reward=0
        T=0
        reward_history=[]
        test_reward_history=[]

        start_time = time.time()
        for e in range(len(episodes)):
            # TODO: Reset environment
            s=self.env.reset()
            episodes.set_description("last episode time {t:d}, last total reward {tr:f}".format(t=t,tr=total_reward))
            episodes.update()
            total_reward=0

            for step in range(len(steps)):
                # TODO: Select action (exercise 2.2)
                a,logp=self.actor.forward(s)
                #v=self.critic.forward(s)
                sp, r, terminal, truncated, info = self.env.step(2*a)
                self.DATA.add(s, a, r, sp, (terminal or truncated), logp=logp)
                s=sp

                steps.set_description("step {t:d}, exploration {ef:f}, mean loss {l:f}".format(t=step,ef=0.99**e,l=total_reward))
                steps.update()
                total_reward=total_reward+r
                if terminal or truncated:
                    reward_history.append(total_reward)
                    if e+1==3:
                        end_time = time.time()

                    # Advantage computation
                    ep_indx_1= 0 if len(np.where(self.DATA.terminal==1.0)[0])==1 else np.where(self.DATA.terminal==1.0)[0][-2]+1
                    ep_indx_2= np.where(self.DATA.terminal==1.0)[0][-1] + 1 if len(np.where(self.DATA.terminal==1.0)[0])==1 else np.where(self.DATA.terminal==1.0)[0][-1] + 1
                    
                    # transform memory to tensors of episodes
                    r_tensor=self.transform_data(self.DATA.r,ep_indx_2,int(ep_indx_2-ep_indx_1)) # [eps steps 1]
                    

#
                    slice=np.arange(ep_indx_1,ep_indx_2)
                    #T=ep_indx_2-ep_indx_1
                    ## Gather self.data
                    ##for t in range(T-1,-1,-1):
                    ##    new_adv_batch[t] = GAMMA * LAMBDA * (0 if t+1==T else new_adv_batch[t+1]) + new_delta_batch[t]
                    ##    new_rtg_batch[t] = (self.DATA.r[slice][t]) if t+1==T else (GAMMA * new_rtg_batch[t+1] + self.DATA.r[slice][t])
#
                    #new_adv_batch[t] = GAMMA * LAMBDA * (0 if t+1==T else new_adv_batch[t+1]) + new_delta_batch[t]
#
                    #self.DATA.adv[slice]=new_adv_batch
                    #self.DATA.rtg[slice]=new_rtg_batch


                    break
            if (e+1)%10==0 and e>0:
                for epoch in range(EPOCHS):
                    rand_idx=np.random.choice(np.arange(len(self.DATA)),len(self.DATA))
                    s_batch,a_p_batch,logp_batch,adv_batch,rtg_batch=self.DATA.s[rand_idx],self.DATA.a[rand_idx],self.DATA.logp[rand_idx],self.DATA.adv[rand_idx],self.DATA.rtg[rand_idx]
                    self.actor.update(s_batch,a_p_batch,logp_batch,adv_batch)

                #for epoch in range(EPOCHS):
                #    rand_idx=np.random.choice(np.arange(len(self.DATA)),len(self.DATA))
                #    s_batch,a_p_batch,logp_batch,adv_batch,rtg_batch=self.DATA.s[rand_idx],self.DATA.a[rand_idx],self.DATA.logp[rand_idx],self.DATA.adv[rand_idx],self.DATA.rtg[rand_idx]
                #    self.critic.update(s_batch,rtg_batch)

                self.DATA=Memory(*self.mem_args)

        window_size = 5
        learn_time = 1000
        for i in range(len(reward_history) - window_size + 1):
            window = reward_history[i:i+window_size]
            if all(r > -500 for r in window):
                learn_time = i + window_size
                break
            
        for e in range(100):
            # TODO: Reset self.environment
            s=self.env.reset()
            episodes.update()
            total_reward=0

            for t in range(len(steps)):
                a,_=self.actor.forward(s)
                sp, r, terminal, truncated, info = self.env.step(2*a)
                s=sp
                total_reward=total_reward+r
                if terminal or truncated:
                    test_reward_history.append(total_reward)
                    plt.plot(reward_history)
                    break

        self.env.close()
        return (end_time-start_time),learn_time,test_reward_history,reward_history