import numpy as np
import time
from ele2364 import Memory
from models.Agents import Planner
from models.Agents import EnsembleModel,RewardModel
class ActiveInference_trainer(object):
    def __init__(self, agent,environment,Memory,random_policy,normalizer):
        #parametric model implements trainin loops and reset weights routines and computation of actions
        self.env=environment
        self.DATA=Memory

        self.Actor=agent
        self.normalizer=normalizer
        self.Actor.ensemble.normalizer=normalizer
        #self.random_policy=lambda :np.random.uniform(0,2)
        self.random_policy=random_policy
    def train(self,episodes,max_steps=500):
        reward_history=[]
        times=[]
        
        for ep in range(episodes):
            print(ep)
            start_time = time.time()
            total_reward=0

            s=self.env.reset()
            # Reset models

            for step in range(max_steps):

                if ep!=0:
                    a=self.Actor.forward(s)
                    sp, r, terminal, truncated, info = self.env.step(a)
                else:
                    a=self.random_policy()
                    sp, r, terminal, truncated, info = self.env.step(a)
                self.DATA.add(s, a, r, sp, (terminal or truncated))
                self.normalizer.update(s,a,sp-s)
                s=sp

                total_reward=total_reward+r
                if terminal or truncated:
                    reward_history.append(total_reward)
                    times.append(time.time()-start_time)
                    print(total_reward)

                    # train reward and world models with memory     
                    self.Actor.train(100,self.DATA)                  
                    break

            
