import numpy as np
import matplotlib.pyplot as plt
import gym



# discretize the spaces ... from continuous to discrete (ckeck gym for info)
x_space = np.linspace(-4.8, 4.8, 15)
x_dot_space = np.linspace(-5, 5, 15)
theta_space = np.linspace(-0.42, 0.42, 15)
theta_dot_space = np.linspace(-5, 5, 15)


# function that gives me the final discrete states each time the new
# states are observed
def get_State(observation):
    x, x_dot, theta, theta_dot = observation
    x = np.digitize(x, x_space)
    x_dot = np.digitize(x_dot, x_dot_space)
    theta = np.digitize(theta, theta_space)
    theta_dot = np.digitize(theta_dot, theta_dot_space)
    return (x, x_dot, theta, theta_dot)

def max_Action(Q, state, actions):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return action


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ALPHA = 0.1
    GAMMA = 0.998
    EPSILON = 1.0

    actions = [0, 1]

    # construct state-space (every possible combination)
    states = []
    for i in range(len(x_space)+1):
        for j in range(len(x_dot_space)+1):
            for k in range(len(theta_space)+1):
                for l in range(len(theta_dot_space)+1):
                    states.append((i, j, k, l))

    # construct & initialize Q table
    Q = {}
    for s in states:
        for a in actions:
            Q[(s, a)] = 0.0

    EPISODES = 15_000
    total_Rewards = np.zeros(EPISODES)
    Rewards = 0
    for i in range(EPISODES):
        if i % 1_000 == 0:
            print('episode', i, 'reward', Rewards, 'EPS', EPSILON)
        observation = env.reset()
        s = get_State(observation)
        done = False
        Rewards = 0
        while not done:
            rand = np.random.random()
            if rand < EPSILON:
                a = env.action_space.sample() # a = np.random.choice(actions)
            else:
                a = max_Action(Q, s, actions)
            observation_, reward, done, info = env.step(a)
            s_ = get_State(observation_)
            a_ = max_Action(Q, s_, actions)
            Rewards += reward
            Q[s, a] = Q[s, a] + ALPHA*(reward + GAMMA*Q[s_, a_] - Q[s, a])
            s = s_
        if EPSILON > 0.01:
            EPSILON -= 2/EPISODES
        else:
            EPSILON = 0.01
        total_Rewards[i] = Rewards

    mean_rewards = np.zeros(EPISODES)
    for t in range(EPISODES):
        mean_rewards[t] = np.mean(total_Rewards[max(0, t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.show()
