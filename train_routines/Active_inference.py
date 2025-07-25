"""Training loop for the Active Inference style agent."""

import numpy as np
import time
from ele2364 import Memory
from models.Agents import Planner
from models.Agents import EnsembleModel,RewardModel
from tqdm.auto import tqdm

class ActiveInference_trainer(object):
    """Trainer for the planner-based Active Inference agent.

    It alternates between collecting data using a model predictive
    controller and training an ensemble dynamics model together with a
    reward predictor. The resulting planner is then used to roll out new
    trajectories, allowing the agent to gradually improve its internal
    world model.

    Parameters
    ----------
    agent : Planner
        Agent implementing planning and action selection.
    environment : gym.Env
        Environment from which transitions are collected.
    Memory : Memory
        Replay buffer used to store data.
    random_policy : Callable
        Function returning random actions for exploration.
    normalizer : callable
        Online normalizer applied to observations.
    model_learning_epochs : int
        Number of training epochs for the ensemble model per episode.
    """
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
    def train(self,episodes,test_episodes=5):
        
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
            

            #s=self.env.reset()
            s=self.env.reset()[0] if isinstance(self.env.reset(),tuple) else self.env.reset()
            episodes.set_description(
                "last episode time {t:d}, last total reward {tr:f}, R error {re:f}, state error {se:f}".format(t=t,tr=total_reward,re=reward_loss,se=s_e_loss))
            episodes.update()
            total_reward=0

            for step in range(len(steps)):

                if e!=0:
                    a=np.round(np.clip(self.Actor.forward(s),0,2)).astype(int)
                    #print(np.round(np.clip(a,0,2)).astype(int))
                    #sp, r, terminal, truncated, info = self.env.step(a)
                else:
                    #a=self.random_policy()
                    a=np.round(np.clip(self.random_policy(),0,2)).astype(int)
                #print(a)
                sp, r, terminal, truncated, info = self.env.step(a[0])
                self.DATA.add(s, a, r, sp, (terminal or truncated))
                self.normalizer.update(s,a,sp-s)
                s=sp

                #steps.set_description("step {t:d}, exploration {ef:f}, mean loss {l:f}".format(t=step,ef=0.99**e,l=total_reward))

                total_reward=total_reward+r
                if terminal or truncated:
                    if e==5:
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

        test_episodes=tqdm(range(test_episodes))
        for e in range(len(test_episodes)):
            # TODO: Reset environment
            #s=self.env.reset()
            s=self.env.reset()[0] if isinstance(self.env.reset(),tuple) else self.env.reset()
            episodes.update()
            total_reward=0

            for t in range(len(steps)):
                #a=self.random_policy()
                a=np.round(np.clip(self.Actor.forward(s),0,2)).astype(int)
                sp, r, terminal, truncated, info = self.env.step(a[0])
                s=sp
                total_reward=total_reward+r
                if terminal or truncated:
                    test_reward_history.append(total_reward)
                    break

        self.env.close()
        return (end_time-start_time),learn_time,test_reward_history,reward_history,reward_error_history,state_error_history

            
