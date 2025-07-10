
from random import choice
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ele2364.networks import Mu, DQ,ttf
import numpy as np
import time

#EPOCHS=80
#BATCH_SIZE=200
#GAMMA=0.99
#LAMBDA=0.97

class DDPG(object):

    def __init__(self,actor,critic,actor_target,critic_target,gamma,environment,Memory,model_learning_epochs):
        self.env=environment
        self.actor=actor
        self.actor_target=actor_target
        self.critic=critic
        self.critic_target=critic_target
        self.DATA=Memory
        self.model_learning_epochs=model_learning_epochs
        self.gamma=gamma

    def train(self,episodes):
        self.Critic_target.copyfrom(self.Critic)
        self.Actor_target.copyfrom(self.Actor)

        episodes=tqdm(range(episodes))
        steps=tqdm(range(1000))
        t=0
        total_reward=0
        T=0
        reward_history=[]
        test_reward_history=[]

        start_time = time.time()
        for e in range(len(episodes)):
            # TODO: Reset self.environment
            s=self.env.reset()
            episodes.set_description("last episode time {t:d}, last total reward {tr:f}".format(t=t,tr=total_reward))
            episodes.update()
            total_reward=0

            for step in range(len(steps)):
                # TODO: Select action (exercise 2.2)
                a=self.Actor.forward(s)
                sp, r, terminal, truncated, info = self.env.step(np.clip(2*a+np.random.randn(1)*0.2,-2,2))
                self.DATA.add(s, a, r, sp, (terminal or truncated))
                s=sp

                steps.set_description("step {t:d}, exploration {ef:f}, mean loss {l:f}".format(t=step,ef=0.99**e,l=total_reward))
                steps.update()
                total_reward=total_reward+r

                s_batch, a_batch, r_batch, sp_batch, done_batch = self.DATA.sample(200)
                # Update Critic
                y= r_batch + (1 - done_batch) * self.gamma * self.Critic_target.forward(sp_batch,self.Actor_target.forward(sp_batch))

                self.Critic.update(s=s_batch,a=a_batch,targets=y)
                    # Update Actor
                self.Actor.update(s_batch,self.Critic)
                #self.Critic_target.copyfrom(Critic)
                #self.Actor_target.copyfrom(Actor)
                #Soft update
                for target_param, param in zip(self.Critic_target.parameters(), self.Critic.parameters()):
                    target_param.data.copy_(target_param.data + 0.01 * (param.data - target_param.data))

                for target_param, param in zip(self.Actor_target.parameters(), self.Actor.parameters()):
                    target_param.data.copy_(target_param.data + 0.01 * (param.data - target_param.data))


                if terminal or truncated:
                    if e+1==100:
                        end_time = time.time()
                    reward_history.append(total_reward)

                    break

        window_size = 5
        learn_time = 150
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
                a=Actor.forward(s)
                sp, r, terminal, truncated, info = self.env.step(2*a)
                s=sp
                total_reward=total_reward+r
                if terminal or truncated:
                    test_reward_history.append(total_reward)
                    break
        # TODO: Close self.environment
        self.env.close()
        return (end_time-start_time),learn_time,test_reward_history,reward_history,Critic,Actor
