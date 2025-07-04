

from random import choice
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from ele2364.networks import Pi, V

EPOCHS=80
BATCH_SIZE=2000
GAMMA=0.99
LAMBDA=0.97

def PPO():
    env=Pendulum()

    # TODO: Create network
    Actor=Pi(3,1,lr=3e-4)
    Critic=V(3,lr=1e-3)
    # TODO: Create replay memory
    DATA=Memory(3,1)


    episodes=tqdm(range(1000))
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
            a,logp=Actor.forward(s)
            v=Critic.forward(s)
            sp, r, terminal, truncated, info = env.step(2*a)
            DATA.add(s, a, r, sp, (terminal or truncated), v=v, logp=logp)
            s=sp

            steps.set_description("step {t:d}, exploration {ef:f}, mean loss {l:f}".format(t=step,ef=0.99**e,l=total_reward))
            steps.update()
            total_reward=total_reward+r
            if terminal or truncated:
                reward_history.append(total_reward)
                if e+1==100:
                    end_time = time.time()

                # Generalized Advantage computation
                ep_indx_1= 0 if len(np.where(DATA.terminal==1.0)[0])==1 else np.where(DATA.terminal==1.0)[0][-2]+1
                ep_indx_2= np.where(DATA.terminal==1.0)[0][-1] + 1 if len(np.where(DATA.terminal==1.0)[0])==1 else np.where(DATA.terminal==1.0)[0][-1] + 1
                new_delta_batch,new_adv_batch,new_rtg_batch=np.zeros((ep_indx_2-ep_indx_1,1)),np.zeros((ep_indx_2-ep_indx_1,1)),np.zeros((ep_indx_2-ep_indx_1,1))

                slice=np.arange(ep_indx_1,ep_indx_2)
                T=ep_indx_2-ep_indx_1
                # Gather data
                for t in range(T-1,-1,-1):
                    new_delta_batch[t] = DATA.r[slice][t] + GAMMA * (0 if t+1==T else DATA.v[slice][t+1]) - DATA.v[slice][t]
                    new_adv_batch[t] = GAMMA * LAMBDA * (0 if t+1==T else new_adv_batch[t+1]) + new_delta_batch[t]
                    new_rtg_batch[t] = (DATA.r[slice][t]) if t+1==T else (GAMMA * new_rtg_batch[t+1] + DATA.r[slice][t])

                DATA.adv[slice]=new_adv_batch
                DATA.rtg[slice]=new_rtg_batch


                break
        if (e+1)%10==0 and e>0:
            for epoch in range(EPOCHS):
                rand_idx=np.random.choice(np.arange(len(DATA)),len(DATA))
                s_batch,a_p_batch,logp_batch,adv_batch,rtg_batch=DATA.s[rand_idx],DATA.a[rand_idx],DATA.logp[rand_idx],DATA.adv[rand_idx],DATA.rtg[rand_idx]
                Actor.update(s_batch,a_p_batch,logp_batch,adv_batch)

            for epoch in range(EPOCHS):
                rand_idx=np.random.choice(np.arange(len(DATA)),len(DATA))
                s_batch,a_p_batch,logp_batch,adv_batch,rtg_batch=DATA.s[rand_idx],DATA.a[rand_idx],DATA.logp[rand_idx],DATA.adv[rand_idx],DATA.rtg[rand_idx]
                Critic.update(s_batch,rtg_batch)
            DATA=Memory(3,1)

    window_size = 5
    learn_time = 1000
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
            a,_=Actor.forward(s)
            sp, r, terminal, truncated, info = env.step(2*a)
            s=sp
            total_reward=total_reward+r
            if terminal or truncated:
                test_reward_history.append(total_reward)
                plt.plot(reward_history)
                break

    env.close()
    return (end_time-start_time),learn_time,test_reward_history,reward_history,Critic,Actor