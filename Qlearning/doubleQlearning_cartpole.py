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

def max_Action(Q1, Q2, state):
    values = np.array([Q1[state, a] + Q2[state, a] for a in range(2)]) #two actions .. left-right
    action = np.argmax(values)
    return action

# def plotRunningAverage(total_Rewards):
#     N = len(total_Rewards)
#     running_avg = np.empty(N)
#     for t in range(N):
# 	    running_avg[t] = np.mean(total_Rewards[max(0, t-100):(t+1)])
#     plt.plot(running_avg)
#     plt.title("Running Average")
#     plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ALPHA = 0.1
    GAMMA = 0.998
    EPSILON = 1.0

    # construct state-space (every possible combination)
    states = []
    for i in range(len(x_space)+1):
        for j in range(len(x_dot_space)+1):
            for k in range(len(theta_space)+1):
                for l in range(len(theta_dot_space)+1):
                    states.append((i, j, k, l))

    Q1 = {}
    Q2 = {}
    for s in states:
        for a in range(2):
            Q1[s, a] = 0
            Q2[s, a] = 0

    EPISODES = 50_000
    total_Rewards = np.zeros(EPISODES)
    Rewards = 0
    for i in range(EPISODES):
        if i % 1000 == 0:
            print('episode', i, 'reward', Rewards, 'EPS', EPSILON)
        done = False
        Rewards = 0
        observation = env.reset()
        while not done:
            s = get_State(observation)
            rand = np.random.random()
            a = max_Action(Q1, Q2, s) if rand < (1-EPSILON) else env.action_space.sample()
            observation_, reward, done, info = env.step(a)
            Rewards += reward
            s_ = get_State(observation_)
            rand = np.random.random()
            if rand <= 0.5:
                a_ = max_Action(Q1, Q1, s_)
                Q1[s, a] = Q1[s, a] + ALPHA*(reward + GAMMA*Q2[s_, a_] - Q1[s, a])
            elif rand > 0.5:
                a_ = max_Action(Q2, Q2, s_)
                Q2[s, a] = Q2[s, a] + ALPHA*(reward + GAMMA*Q1[s_, a_] - Q2[s, a])

            observation = observation_
        EPSILON -= 2/(EPISODES) if EPSILON > 0 else 0
        total_Rewards[i] = Rewards

    mean_rewards = np.zeros(EPISODES)
    for t in range(EPISODES):
        mean_rewards[t] = np.mean(total_Rewards[max(0, t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.show()

    #plotRunningAverage(total_Rewards)
