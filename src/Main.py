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


def policy_eval(policy):
    # Initialize expected values to zero
    V = np.zeros(ns)
    # hard code the terminal state values
    V[9] = 1
    V[10] = -1
    while True:
        # Keep track of the update done in value function
        delta = 0
        for s in range(ns - 2):  # -2 because we don't consider the terminal states
            v = 0
            # What the action is for a given state
            a = policy[s]
            # For each action, look at the possible next states,
            for prob, next_state, reward, done in P[s][a]:  # state transition P[s][a] == [(prob, nextstate, reward, done), ...]
                # Calculate the expected value function
                v += prob * (reward + V[next_state])  # P[s, a, s']*(R(s,a,s')+γV[s'])
            # How much our value function changed across any states .
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function update is below a threshold
        if delta < 0.0000000000001:
            break
    return np.array(V)


def policy_improvement(policy):
    while True:
        V = policy_eval(policy)
        policy_stable = True
        for s in range(ns):
            chosen_a = policy[s]
            action_values = np.zeros(na)
            for a in range(na):
                for prob, next_state, reward, done in P[s][a]:
                    action_values[a] += prob * (reward + V[next_state])
            best_a = np.argmax(action_values)

            if chosen_a != best_a:
                policy_stable = False
            policy[s] = best_a
        if policy_stable:
            return policy


def print_policy_value(policy):
    """
    Prints the policy value given the policy value
    :param pvalue: list of expected values given a
    :return:
    """
    print("Policy value, using states as defined by Professor Si:")
    pvalue = policy_eval(policy)
    directions = {0: "N",
                  1: "S",
                  2: "E",
                  3: "W"}
    for i in range(len(pvalue) - 2):
        print("State %d: Action %s: Expected Value: %3.7f" % (i+1, directions[policy[i]], pvalue[i]))
    print("")


if __name__ == "__main__":
    # Initial policy with state 6 south
    print("Evaluating policy given by Professor Si")
    initial_policy = [0, 0, 2, 2, 2, 1, 3, 3, 3, 0, 0]
    print_policy_value(initial_policy)

    # Perform policy iteration
    print("Performing policy iteration")
    optimal_policy = policy_improvement(initial_policy)
    print_policy_value(optimal_policy)

    # Another feasible policy
    print("Evaluating a north-only policy")
    north_policy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print_policy_value(north_policy)
    print("Improve the north-only policy")
    optimal_policy = policy_improvement(north_policy)
    print_policy_value(optimal_policy)
