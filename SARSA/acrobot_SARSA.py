import numpy as np
import matplotlib.pyplot as plt
import gym

# set the state spac from continuous to discrete
theta_space = np.linspace(-1, 1, 10)
theta_dot_space = np.linspace(-5, 5, 10)

def get_State(observation):
    cos_theta1, sin_theta1, cos_theta2 ,sin_theta2, theta1_dot, theta2_dot = observation
    c_th1 = int(np.digitize(cos_theta1, theta_space))
    s_th1 = int(np.digitize(sin_theta1, theta_space))
    c_th2 = int(np.digitize(cos_theta2, theta_space))
    s_th2 = int(np.digitize(sin_theta2, theta_space))
    th1_dot = int(np.digitize(theta1_dot, theta_dot_space))
    th2_dot = int(np.digitize(theta2_dot, theta_dot_space))

    return (c_th1, s_th1, c_th2, s_th2, th1_dot, th2_dot)

def max_Action(Q, state):
    values = np.array([Q[state, a] for a in range(3)])
    action = np.argmax(values)
    return action



if __name__ == '__main__':
    env = gym.make('Acrobot-v1')

    ALPHA = 0.1
    GAMMA = 0.99
    EPSILON = 1.0
    load = False

    states = []
    for c1 in range(11):
        for s1 in range(11):
            for c2 in range(11):
                for s2 in range(11):
                    for dot1 in range(11):
                        for dot2 in range(11):
                            states.append((c1, s1, c2, s2, dot1, dot2))

    if load == False:
        Q = {}
        for s in states:
            for a in range(3):
                Q[s, a] = 0
    else:
        pickle_in = ('acrobot.pkl', 'rb')
        Q = pickle.load(pickle_in)


    EPISODES = 50_000
    totalRewards = np.zeros(EPISODES)
    Rewards = 0
    for i in range(EPISODES):
        if i % 1_000 == 0:
            print('episode ', i, 'reward ', Rewards, 'eps', EPSILON)
        observation = env.reset()
        s = get_State(observation)  # current state S
        rand = np.random.random()
        # ε-greedy action selection
        if rand < (1-EPSILON):
            a = max_Action(Q, s)  # current action A I will take
        else:
            a = env.action_space.sample()  # random action A
        done = False
        Rewards = 0
        while not done:
            observation_, reward, done, info = env.step(a)  # take that action and observe next state and reward
            s_ = get_State(observation_) # next state S'
            rand = np.random.random()
            if rand > EPSILON:
                a_ = max_Action(Q, s_)  # next action A'
            else:
                a_ = env.action_space.sample()  # random action A'
            Rewards += reward
            Q[s, a] = Q[s, a] + ALPHA*(reward + GAMMA*Q[s_, a_] - Q[s, a])
            s, a = s_, a_
        EPSILON -= 2/(EPISODES) if EPSILON > 0.01 else 0.01
        totalRewards[i] = Rewards

    mean_rewards = np.zeros(EPISODES)
    for t in range(EPISODES):
        mean_rewards[t] = np.mean(total_Rewards[max(0, t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.show()

    f = open("acrobot.pkl","wb")
    pickle.dump(Q,f)
    f.close()  # save the file if I want to get back and continue training
