import numpy as np
import time
from ele2364 import Memory
from models.Agents import Planner
from models.Agents import EnsembleModel,RewardModel
from tqdm.auto import tqdm

class ActiveInference_trainer(object):
    def __init__(self, agent,environment,Memory,random_policy,normalizer,model_learning_epochs):
        #parametric model implements trainin loops and reset weights routines and computation of actions
        self.env=environment
        self.DATA=Memory

        self.Actor=agent
        self.normalizer=normalizer
        self.Actor.ensemble.normalizer=normalizer
        self.model_learning_epochs=model_learning_epochs
        #self.random_policy=lambda :np.random.uniform(0,2)
        self.random_policy=random_policy
    def train(self,episodes,max_steps=500):
        
        episodes=tqdm(range(episodes))
        steps=tqdm(range(1000))
        t=0
        total_reward=0
        T=0
        reward_history=[]
        reward_error_history=[]
        state_error_history=[]
        test_reward_history=[]
        start_time = time.time()

        reward_loss=0
        s_e_loss=0
        
        for e in range(len(episodes)):
            start_time = time.time()
            total_reward=0

            s=self.env.reset()
            episodes.set_description(
                "last episode time {t:d}, last total reward {tr:f}, R error {re:f}, state error {se:f}".format(t=t,tr=total_reward,re=reward_loss,se=s_e_loss))
            episodes.update()

            for step in range(len(steps)):

                if e!=0:
                    a=self.Actor.forward(s)
                    sp, r, terminal, truncated, info = self.env.step(a)
                else:
                    a=self.random_policy()
                    sp, r, terminal, truncated, info = self.env.step(a)
                self.DATA.add(s, a, r, sp, (terminal or truncated))
                self.normalizer.update(s,a,sp-s)
                s=sp

                #steps.set_description("step {t:d}, exploration {ef:f}, mean loss {l:f}".format(t=step,ef=0.99**e,l=total_reward))

                total_reward=total_reward+r
                if terminal or truncated:
                    if e+1==2:
                        end_time = time.time()

                    reward_history.append(total_reward)
                    
                    #print(total_reward)

                    # train reward and world models with memory     
                    s_e_loss,reward_loss=self.Actor.train(self.model_learning_epochs,self.DATA)
                    reward_error_history.append(reward_loss)
                    state_error_history.append(s_e_loss)
                    #print("state prediction loss, reward prediction loss")
                    #print(s_e_loss,reward_loss)
                    break

        window_size = 5
        learn_time = 150
        for i in range(len(reward_history) - window_size + 1):
            window = reward_history[i:i+window_size]
            if all(r > -500 for r in window):
                learn_time = i + window_size
                break

        for e in range(50):
            # TODO: Reset environment
            s=self.env.reset()
            episodes.update()
            total_reward=0

            for t in range(len(steps)):
                a=self.random_policy()
                sp, r, terminal, truncated, info = self.env.step(a)
                s=sp
                total_reward=total_reward+r
                if terminal or truncated:
                    test_reward_history.append(total_reward)
                    break

        self.env.close()
        return (end_time-start_time),learn_time,test_reward_history,reward_history,reward_error_history,state_error_history

            
