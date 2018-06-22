import numpy as np
import matplotlib.pyplot as plt
import time

def epsilon_greedy(s, epsilon = 0.1, m = 8):
   sorted_index = np.argsort(s[0:-1])
   max_index = sorted_index[-1]
   other_index = sorted_index[:-1]
   max_prob = epsilon / m + (1 - epsilon)
   other_prob = epsilon / m
   rm = np.random.rand(1)[0]
   if rm < max_prob:
       return int(max_index)
   else:
        return int(other_index[np.random.randint(0, len(s[0:-1]) - 1)])

'''
def next_s(action, S, cur_s1, cur_s2):
    next_s1, next_s2 = cur_s1, cur_s2
    if action == 'left':
        next_s2 = cur_s2 - 1
        next_s1 = cur_s1 - S[cur_s1][cur_s2][-1]
        if next_s1 < 0:
            next_s1 = 0
        if next_s2 < 0:
            next_s2 = 0
        return int(next_s1), int(next_s2)
    elif action == 'up':
        next_s1 = cur_s1 - 1 - S[cur_s1][cur_s2][-1]
        if next_s1 < 0:
            next_s1 = 0
        return int(next_s1), int(next_s2)
    elif action == 'right':
        next_s2 = cur_s2 + 1
        next_s1 = cur_s1 - S[cur_s1][cur_s2][-1]
        if next_s1 < 0:
            next_s1 = 0
        if next_s2 >= S.shape[1]:
            next_s2 = S.shape[1] - 1
        return int(next_s1), int(next_s2)
    elif action == 'down':
        next_s1 = cur_s1 + 1 - S[cur_s1][cur_s2][-1]
        if next_s1 >= S.shape[0]:
            next_s1 = S.shape[0] - 1
        elif next_s1 < 0:
            next_s1 = 0
        return int(next_s1), int(next_s2)
    else:
        print("No action!!")
'''

def next_s(action, S, cur_s1, cur_s2):
    next_s1, next_s2 = cur_s1, cur_s2
    if action == 'left':
        next_s2 = cur_s2 - 1
        next_s1 = cur_s1 - S[cur_s1][cur_s2][-1]
        if next_s1 < 0:
            next_s1 = 0
        if next_s2 < 0:
            next_s2 = 0
        return int(next_s1), int(next_s2)
    elif action == 'up':
        next_s1 = cur_s1 - 1 - S[cur_s1][cur_s2][-1]
        if next_s1 < 0:
            next_s1 = 0
        return int(next_s1), int(next_s2)
    elif action == 'right':
        next_s2 = cur_s2 + 1
        next_s1 = cur_s1 - S[cur_s1][cur_s2][-1]
        if next_s1 < 0:
            next_s1 = 0
        if next_s2 >= S.shape[1]:
            next_s2 = S.shape[1] - 1
        return int(next_s1), int(next_s2)
    elif action == 'down':
        next_s1 = cur_s1 + 1 - S[cur_s1][cur_s2][-1]
        if next_s1 >= S.shape[0]:
            next_s1 = S.shape[0] - 1
        elif next_s1 < 0:
            next_s1 = 0
        return int(next_s1), int(next_s2)
    elif action == 'left_up':
        next_s1 = cur_s1 - 1 - S[cur_s1][cur_s2][-1]
        next_s2 = cur_s2 - 1
        if next_s1 < 0:
            next_s1 = 0
        if next_s2 < 0:
            next_s2 = 0
        return int(next_s1), int(next_s2)
    elif action == 'right_up':
        next_s1 = cur_s1 - 1 - S[cur_s1][cur_s2][-1]
        next_s2 = cur_s2 + 1
        if next_s1 < 0:
            next_s1 = 0
        if next_s2 >= S.shape[1]:
            next_s2 = S.shape[1] - 1
        return int(next_s1), int(next_s2)
    elif action == 'right_down':
        next_s1 = cur_s1 + 1 - S[cur_s1][cur_s2][-1]
        next_s2 = cur_s2 + 1
        if next_s1 >= S.shape[0]:
            next_s1 = S.shape[0] - 1
        elif next_s1 < 0:
            next_s1 = 0
        if next_s2 >= S.shape[1]:
            next_s2 = S.shape[1] - 1
        return int(next_s1), int(next_s2)
    elif action == 'left_down':
        next_s1 = cur_s1 + 1 - S[cur_s1][cur_s2][-1]
        next_s2 = cur_s2 - 1
        if next_s1 >= S.shape[0]:
            next_s1 = S.shape[0] - 1
        elif next_s1 < 0:
            next_s1 = 0
        if next_s2 < 0:
            next_s2 = 0
        return int(next_s1), int(next_s2)
    else:
        print("No action!!")

def windyGridWorld(S, actions, alpha = 0.1):
    new_S = S
    episode_length = []
    for i in range(1000):
        steps = []
        cur_s = new_S[3][0]
        cur_s1, cur_s2 = 3, 0
        start = time.time()
        while True:
            cur_action_index = epsilon_greedy(cur_s)
            next_s1, next_s2 = next_s(actions[cur_action_index], S, cur_s1, cur_s2)
            max_q = np.max(S[next_s1][next_s2][0:-1])
            steps.append(actions[cur_action_index])
            new_S[cur_s1][cur_s2][cur_action_index] += alpha * (-1 + max_q - new_S[cur_s1][cur_s2][cur_action_index])
            cur_s1, cur_s2 = next_s1, next_s2
            cur_s = S[next_s1][next_s2]
            if cur_s1 == 3 and cur_s2 == 7:
                break
        stop = time.time()
        print(steps)
        episode_length.append(1000* (stop - start))
    return new_S, episode_length

if __name__ == "__main__":
    ts = time.time()
    actions = ['left', 'left_up', 'up', 'right_up', 'right', 'right_down', 'down', 'left_down',]
    # actions = ['left', 'up', 'right', 'down',]
    S = np.zeros((7, 10, len(actions) + 1))
    S[3][7] = 0  # terminal
    S[:, 3:6, -1] = 1
    S[:, 6:8, -1] = 2
    S[:, 8, -1] = 1
    S, episodes_length = windyGridWorld(S, actions)
    tp = time.time()
    print(1000 * (tp - ts))

    #plot
    plt.figure("Q-learning", (10,5))
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Q-learning----Episode Length over time")
    plt.plot(np.arange(0, 1000), episodes_length, c='b')
    plt.show()
