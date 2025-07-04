%matplotlib inline
from random import choice
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ele2364.networks import Mu, DQ,ttf

EPOCHS=80
BATCH_SIZE=200
GAMMA=0.99
LAMBDA=0.97

def DDPG():
    env=Pendulum()
    Actor=Mu(3,1)
    Actor_target=Mu(3,1)
    Critic=CQ(3,1)
    Critic_target=CQ(3,1)

    Critic_target.copyfrom(Critic)
    Actor_target.copyfrom(Actor)
    # TODO: Create replay memory
    DATA=Memory(3,1)


    episodes=tqdm(range(150))
    steps=tqdm(range(1000))
    t=0
    total_reward=0
    T=0
    reward_history=[]
    test_reward_history=[]

    start_time = time.time()
    for e in range(len(episodes)):
        # TODO: Reset environment
        s=env.reset()
        episodes.set_description("last episode time {t:d}, last total reward {tr:f}".format(t=t,tr=total_reward))
        episodes.update()
        total_reward=0

        for step in range(len(steps)):
            # TODO: Select action (exercise 2.2)
            a=Actor.forward(s)
            sp, r, terminal, truncated, info = env.step(np.clip(2*a+np.random.randn(1)*0.2,-2,2))
            DATA.add(s, a, r, sp, (terminal or truncated))
            s=sp

            steps.set_description("step {t:d}, exploration {ef:f}, mean loss {l:f}".format(t=step,ef=0.99**e,l=total_reward))
            steps.update()
            total_reward=total_reward+r

            s_batch, a_batch, r_batch, sp_batch, done_batch = DATA.sample(200)
            # Update Critic
            y= r_batch + (1 - done_batch) * GAMMA * Critic_target.forward(sp_batch,Actor_target.forward(sp_batch))

            Critic.update(s=s_batch,a=a_batch,targets=y)
                # Update Actor
            Actor.update(s_batch,Critic)
            #Critic_target.copyfrom(Critic)
            #Actor_target.copyfrom(Actor)
            #Soft update
            for target_param, param in zip(Critic_target.parameters(), Critic.parameters()):
                target_param.data.copy_(target_param.data + 0.01 * (param.data - target_param.data))

            for target_param, param in zip(Actor_target.parameters(), Actor.parameters()):
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
        # TODO: Reset environment
        s=env.reset()
        episodes.update()
        total_reward=0

        for t in range(len(steps)):
            a=Actor.forward(s)
            sp, r, terminal, truncated, info = env.step(2*a)
            s=sp
            total_reward=total_reward+r
            if terminal or truncated:
                test_reward_history.append(total_reward)
                break
    # TODO: Close environment
    env.close()
    return (end_time-start_time),learn_time,test_reward_history,reward_history,Critic,Actor
