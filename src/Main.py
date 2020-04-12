import numpy as np

ns = 11  # Number of states
na = 4  # Number of actions
P = [[[] for x in range(na)] for y in range(ns)]

# state-action probability transition matrix
# [state][action]
# states go from 0 to 8
# action 0 is North, 1 is South, 2 is East, 3 is West
P[0][0] = [(0.8, 1, -0.04, False),
           (0.1, 0, -0.04, False),
           (0.1, 8, -0.04, False)]
P[0][1] = [(0.8, 0, -0.04, False),
           (0.1, 8, -0.04, False),
           (0.1, 0, -0.04, False)]
P[0][2] = [(0.8, 8, -0.04, False),
           (0.1, 1, -0.04, False),
           (0.1, 0, -0.04, False)]
P[0][3] = [(0.8, 0, -0.04, False),
           (0.1, 0, -0.04, False),
           (0.1, 1, -0.04, False)]

P[1][0] = [(0.8, 2, -0.04, False),
           (0.1, 1, -0.04, False),
           (0.1, 1, -0.04, False)]
P[1][1] = [(0.8, 0, -0.04, False),
           (0.1, 1, -0.04, False),
           (0.1, 1, -0.04, False)]
P[1][2] = [(0.8, 1, -0.04, False),
           (0.1, 2, -0.04, False),
           (0.1, 0, -0.04, False)]
P[1][3] = [(0.8, 1, -0.04, False),
           (0.1, 0, -0.04, False),
           (0.1, 2, -0.04, False)]

P[2][0] = [(0.8, 2, -0.04, False),
           (0.1, 2, -0.04, False),
           (0.1, 3, -0.04, False)]
P[2][1] = [(0.8, 1, -0.04, False),
           (0.1, 3, -0.04, False),
           (0.1, 2, -0.04, False)]
P[2][2] = [(0.8, 3, -0.04, False),
           (0.1, 2, -0.04, False),
           (0.1, 1, -0.04, False)]
P[2][3] = [(0.8, 2, -0.04, False),
           (0.1, 1, -0.04, False),
           (0.1, 2, -0.04, False)]

P[3][0] = [(0.8, 3, -0.04, False),
           (0.1, 2, -0.04, False),
           (0.1, 4, -0.04, False)]
P[3][1] = [(0.8, 3, -0.04, False),
           (0.1, 4, -0.04, False),
           (0.1, 2, -0.04, False)]
P[3][2] = [(0.8, 4, -0.04, False),
           (0.1, 3, -0.04, False),
           (0.1, 3, -0.04, False)]
P[3][3] = [(0.8, 2, -0.04, False),
           (0.1, 3, -0.04, False),
           (0.1, 3, -0.04, False)]

P[4][0] = [(0.8, 4, -0.04, False),
           (0.1, 3, -0.04, False),
           (0.1, 9, -0.04, False)]
P[4][1] = [(0.8, 5, -0.04, False),
           (0.1, 9, -0.04, False),
           (0.1, 3, -0.04, False)]
P[4][2] = [(0.8, 9, -0.04, False),
           (0.1, 4, -0.04, False),
           (0.1, 5, -0.04, False)]
P[4][3] = [(0.8, 2, -0.04, False),
           (0.1, 5, -0.04, False),
           (0.1, 4, -0.04, False)]

P[5][0] = [(0.8, 4, -0.04, False),
           (0.1, 5, -0.04, False),
           (0.1, 10, -0.04, False)]
P[5][1] = [(0.8, 7, -0.04, False),
           (0.1, 10, -0.04, False),
           (0.1, 5, -0.04, False)]
P[5][2] = [(0.8, 10, -0.04, False),
           (0.1, 4, -0.04, False),
           (0.1, 7, -0.04, False)]
P[5][3] = [(0.8, 5, -0.04, False),
           (0.1, 7, -0.04, False),
           (0.1, 4, -0.04, False)]

P[6][0] = [(0.8, 10, -0.04, False),
           (0.1, 7, -0.04, False),
           (0.1, 6, -0.04, False)]
P[6][1] = [(0.8, 6, -0.04, False),
           (0.1, 7, -0.04, False),
           (0.1, 6, -0.04, False)]
P[6][2] = [(0.8, 6, -0.04, False),
           (0.1, 10, -0.04, False),
           (0.1, 6, -0.04, False)]
P[6][3] = [(0.8, 7, -0.04, False),
           (0.1, 6, -0.04, False),
           (0.1, 10, -0.04, False)]

P[7][0] = [(0.8, 5, -0.04, False),
           (0.1, 8, -0.04, False),
           (0.1, 6, -0.04, False)]
P[7][1] = [(0.8, 7, -0.04, False),
           (0.1, 6, -0.04, False),
           (0.1, 8, -0.04, False)]
P[7][2] = [(0.8, 6, -0.04, False),
           (0.1, 5, -0.04, False),
           (0.1, 8, -0.04, False)]
P[7][3] = [(0.8, 8, -0.04, False),
           (0.1, 7, -0.04, False),
           (0.1, 5, -0.04, False)]

P[8][0] = [(0.8, 8, -0.04, False),
           (0.1, 0, -0.04, False),
           (0.1, 7, -0.04, False)]
P[8][1] = [(0.8, 8, -0.04, False),
           (0.1, 7, -0.04, False),
           (0.1, 0, -0.04, False)]
P[8][2] = [(0.8, 7, -0.04, False),
           (0.1, 8, -0.04, False),
           (0.1, 8, -0.04, False)]
P[8][3] = [(0.8, 0, -0.04, False),
           (0.1, 8, -0.04, False),
           (0.1, 8, -0.04, False)]

P[9][0] = [(1, 9, 1, True)]
P[9][1] = [(1, 9, 1, True)]
P[9][2] = [(1, 9, 1, True)]
P[9][3] = [(1, 9, 1, True)]

P[10][0] = [(1, 10, -1, True)]
P[10][1] = [(1, 10, -1, True)]
P[10][2] = [(1, 10, -1, True)]
P[10][3] = [(1, 10, -1, True)]


# policy = [(0, 1),
#           (0, 1),
#           (2, 1),
#           (2, 1),
#           (2, 1),
#           (0, 1),
#           (3, 1),
#           (3, 1),
#           (3, 1)]

policy = [0, 0, 2, 2, 2, 1, 3, 3, 3, 0, 0]

def policy_eval():
    # Initialize thel value function
    V = np.zeros(ns)
    V[9] = 1
    V[10] = -1
    # While our value function is worse than the threshold theta
    while True:
        # Keep track of the update done in value function
        delta = 0
        # For each state, look ahead one step at each possible action and next state
        for s in range(ns - 2):
            v = 0
            # print(s)
            # The possible next actions, policy[s]:[a,action_prob]
            a = policy[s]
            # For each action, look at the possible next states,
            # print(P[s][a])
            for prob, next_state, reward, done in P[s][a]:  # state transition P[s][a] == [(prob, nextstate, reward, done), ...]
                # Calculate the expected value function
                v += prob * (reward + V[next_state])  # P[s, a, s']*(R(s,a,s')+γV[s'])
                # print(v)
                # How much our value function changed across any states .
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
            print("s: " + str(s) + "\tV[s]: " + str(V[s]))
        # Stop evaluating once our value function update is below a threshold
        if delta < 0.00001:
            break
    return np.array(V)


if __name__ == "__main__":
    arr = policy_eval()
