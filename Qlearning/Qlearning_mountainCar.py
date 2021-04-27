import numpy as np
import matplotlib.pyplot as plt
import gym

# discretize the space ... from continuous to discrete (ckeck gym for info)
pos_space = np.linspace(-1.2, 0.6, 20)
vel_space = np.linspace(-0.07, 0.07, 20)

# function that gives me the final discrete states each time the new
# states are observed
def get_State(observation):
    position, velocity = observation
    position = int(np.digitize(position, pos_space))
    velocity = int(np.digitize(velocity, vel_space))
    return (position, velocity)

# get the max action of the corresponding state
def max_Action(Q, state):
    values = np.array([Q[state, a] for a in range(3)])  # 3 actions
    action = np.argmax(values)
    return action


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    #env._max_episode_steps = 1000
    ALPHA = 0.1    # learning rate
    GAMMA = 0.998  # discount factor
    EPSILON = 1.0  # epsilon for ε-greedy

    # construct state-space (every possible combination)
    states = []
    for p in range(len(pos_space)+1):
        for v in range(len(vel_space)+1):
            states.append((p, v))

    # construct & initialize Q table
    Q = {}
    for s in states:
        for a in range(3):  # 3 actions
            Q[(s, a)] = 0


    EPISODES = 20_000
    total_Rewards = np.zeros(EPISODES)
    Rewards = 0
    for i in range(EPISODES):
        if i % 1_000 == 0:   # show me every 1000 EPISODES
            print('episode', i, 'reward', Rewards, 'epsilon', EPSILON)
        observation = env.reset()
        s = get_State(observation)  # current state S
        done = False
        Rewards = 0
        while not done:
            rand = np.random.random()
            if rand > EPSILON:   # ε-greedy functionality for action A
                a = max_Action(Q, s)
            else:
                a = env.action_space.sample()
            observation_, reward, done, info = env.step(a)  # take action A and observe S', reward
            s_ = get_State(observation_)  # next state S'
            Rewards += reward
            a_ = max_Action(Q, s_)  # next max action A'
            Q[s, a] = Q[s, a] + ALPHA*(reward + GAMMA*Q[s_, a_] - Q[s, a])  #Qlearning update
            s = s_  # set S <-- S'
        EPSILON -= 2/(EPISODES) if EPSILON > 0.01 else 0.01  # linear decay of EPSILON
        total_Rewards[i] = Rewards

    mean_rewards = np.zeros(EPISODES)
    for t in range(EPISODES):
        mean_rewards[t] = np.mean(total_Rewards[max(0, t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.show()
