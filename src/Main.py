#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

ns = 11  # Number of states
na = 4  # Number of actions
P = [[[] for x in range(na)] for y in range(ns)]
Q_values = np.zeros((ns, na))
directions = {0: "N",
              1: "S",
              2: "E",
              3: "W"}

# state-action probability transition matrix
# [state][action]
# states go from 0 to 8
# action 0 is North, 1 is South, 2 is East, 3 is West
P[0][0] = [(0.8, 1, -0.2, False),
           (0.1, 0, -0.2, False),
           (0.1, 8, -0.2, False)]
P[0][1] = [(0.8, 0, -0.2, False),
           (0.1, 8, -0.2, False),
           (0.1, 0, -0.2, False)]
P[0][2] = [(0.8, 8, -0.2, False),
           (0.1, 1, -0.2, False),
           (0.1, 0, -0.2, False)]
P[0][3] = [(0.8, 0, -0.2, False),
           (0.1, 0, -0.2, False),
           (0.1, 1, -0.2, False)]

P[1][0] = [(0.8, 2, -0.2, False),
           (0.1, 1, -0.2, False),
           (0.1, 1, -0.2, False)]
P[1][1] = [(0.8, 0, -0.2, False),
           (0.1, 1, -0.2, False),
           (0.1, 1, -0.2, False)]
P[1][2] = [(0.8, 1, -0.2, False),
           (0.1, 2, -0.2, False),
           (0.1, 0, -0.2, False)]
P[1][3] = [(0.8, 1, -0.2, False),
           (0.1, 0, -0.2, False),
           (0.1, 2, -0.2, False)]

P[2][0] = [(0.8, 2, -0.2, False),
           (0.1, 2, -0.2, False),
           (0.1, 3, -0.2, False)]
P[2][1] = [(0.8, 1, -0.2, False),
           (0.1, 3, -0.2, False),
           (0.1, 2, -0.2, False)]
P[2][2] = [(0.8, 3, -0.2, False),
           (0.1, 2, -0.2, False),
           (0.1, 1, -0.2, False)]
P[2][3] = [(0.8, 2, -0.2, False),
           (0.1, 1, -0.2, False),
           (0.1, 2, -0.2, False)]

P[3][0] = [(0.8, 3, -0.2, False),
           (0.1, 2, -0.2, False),
           (0.1, 4, -0.2, False)]
P[3][1] = [(0.8, 3, -0.2, False),
           (0.1, 4, -0.2, False),
           (0.1, 2, -0.2, False)]
P[3][2] = [(0.8, 4, -0.2, False),
           (0.1, 3, -0.2, False),
           (0.1, 3, -0.2, False)]
P[3][3] = [(0.8, 2, -0.2, False),
           (0.1, 3, -0.2, False),
           (0.1, 3, -0.2, False)]

P[4][0] = [(0.8, 4, -0.2, False),
           (0.1, 3, -0.2, False),
           (0.1, 9, -0.2, False)]
P[4][1] = [(0.8, 5, -0.2, False),
           (0.1, 9, -0.2, False),
           (0.1, 3, -0.2, False)]
P[4][2] = [(0.8, 9, -0.2, False),
           (0.1, 4, -0.2, False),
           (0.1, 5, -0.2, False)]
P[4][3] = [(0.8, 2, -0.2, False),
           (0.1, 5, -0.2, False),
           (0.1, 4, -0.2, False)]

P[5][0] = [(0.8, 4, -0.2, False),
           (0.1, 5, -0.2, False),
           (0.1, 10, -0.2, False)]
P[5][1] = [(0.8, 7, -0.2, False),
           (0.1, 10, -0.2, False),
           (0.1, 5, -0.2, False)]
P[5][2] = [(0.8, 10, -0.2, False),
           (0.1, 4, -0.2, False),
           (0.1, 7, -0.2, False)]
P[5][3] = [(0.8, 5, -0.2, False),
           (0.1, 7, -0.2, False),
           (0.1, 4, -0.2, False)]

P[6][0] = [(0.8, 10, -0.2, False),
           (0.1, 7, -0.2, False),
           (0.1, 6, -0.2, False)]
P[6][1] = [(0.8, 6, -0.2, False),
           (0.1, 7, -0.2, False),
           (0.1, 6, -0.2, False)]
P[6][2] = [(0.8, 6, -0.2, False),
           (0.1, 10, -0.2, False),
           (0.1, 6, -0.2, False)]
P[6][3] = [(0.8, 7, -0.2, False),
           (0.1, 6, -0.2, False),
           (0.1, 10, -0.2, False)]

P[7][0] = [(0.8, 5, -0.2, False),
           (0.1, 8, -0.2, False),
           (0.1, 6, -0.2, False)]
P[7][1] = [(0.8, 7, -0.2, False),
           (0.1, 6, -0.2, False),
           (0.1, 8, -0.2, False)]
P[7][2] = [(0.8, 6, -0.2, False),
           (0.1, 5, -0.2, False),
           (0.1, 8, -0.2, False)]
P[7][3] = [(0.8, 8, -0.2, False),
           (0.1, 7, -0.2, False),
           (0.1, 5, -0.2, False)]

P[8][0] = [(0.8, 8, -0.2, False),
           (0.1, 0, -0.2, False),
           (0.1, 7, -0.2, False)]
P[8][1] = [(0.8, 8, -0.2, False),
           (0.1, 7, -0.2, False),
           (0.1, 0, -0.2, False)]
P[8][2] = [(0.8, 7, -0.2, False),
           (0.1, 8, -0.2, False),
           (0.1, 8, -0.2, False)]
P[8][3] = [(0.8, 0, -0.2, False),
           (0.1, 8, -0.2, False),
           (0.1, 8, -0.2, False)]

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


def value_iteration():
    # Look ahead one step at each possible action and next state (full backup)
    def one_step_lookahead(state, V):
        A = np.zeros(na)
        for a in range(na):
            for prob, next_state, reward, done in P[state][a]:
                A[a] += prob * (reward + V[next_state])
        return A

    V = np.zeros(ns)
    V[9] = 1
    V[10] = -1
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(ns-2):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function
            V[s] = best_action_value
            # Check if we can stop
        if delta < 0.00000000001:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([ns, na])
    for s in range(ns-2):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    policy = np.argmax(policy, 1)
    return policy, V


def print_policy_value(policy):
    """
    Prints the policy value given the policy value
    :param pvalue: list of expected values given a
    :return:
    """
    global directions
    print("Policy value, using states as defined by Professor Si:")
    pvalue = policy_eval(policy)
    for i in range(len(pvalue) - 2):
        print("State %d: Action %s: Expected Value: %3.7f" % (i+1, directions[policy[i]], pvalue[i]))
    print("")


def print_policy(policy):
    global directions
    print("Policy, using states as defined by Professor Si:")
    for i in range(len(policy) - 1):
        print("State %d: Action %s" % (i+1, directions[policy[i]]))
    print("")


def q_learn(episodes):
    global Q_values, directions
    alpha = 0.2
    gamma = 1
    exploration_rate_initial = 0.3
    action = None

    for i in range(episodes):
        episode_complete = False
        state = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
        cumulative_reward = 0
        states = []
        # Gradually reduce learning rate
        exploration_rate = exploration_rate_initial - (i/episodes) * exploration_rate_initial
        while True:
            # Choose an action
            # If random number is less than learning rate, then explore randomly
            if np.random.uniform(0, 1) <= exploration_rate:
                # print("Trying something random")
                action = np.random.choice([0, 1, 2, 3])
            # Otherwise choose the action with the maximum reward associated with it
            else:
                # print("Choosing action associated with maximum reward")
                max_state_transition_reward = -1000
                for a in [0, 1, 2, 3]:
                    state_transition_reward = Q_values[state][a]
                    if state_transition_reward > max_state_transition_reward:
                        max_state_transition_reward = state_transition_reward
                        action = a

            # Record what state-action pair the agent took
            states.append([state, action])
            # print("Current state: %d\t Action taken: %c" % (state+1, directions[action]))

            # Determine which state the agent will end up in
            next_state_prob = np.random.uniform(0, 1)
            if next_state_prob <= 0.8:
                state = P[state][action][0][1]
                # print("Went in desired direction")
            elif next_state_prob <= 0.9:
                state = P[state][action][1][1]
                # print("Went left of desired direction")
            else:
                state = P[state][action][2][1]
                # print("Went right of desired direction")
            cumulative_reward -= 0.2    # Every state transition has this penalty
            # print("Next state: %s: Cumulative Reward: %f" % (state+1, cumulative_reward))
            # print("------------------------------------------------------------------")

            # If terminal condition is reached, end the episode
            if P[state][0][0][3]:
                cumulative_reward += P[state][0][0][2]
                # print("Reached terminal state. Total reward: %f" % cumulative_reward)
                episode_complete = True

            # Update the Q-table every step
            for s in reversed(states):
                current_q_value = Q_values[s[0]][s[1]]
                max_future_q_value = max(Q_values[s[0]])
                # Q_values[s[0]][s[1]] = current_q_value + alpha * (cumulative_reward + gamma * max_future_q_value - current_q_value)
                Q_values[s[0]][s[1]] = current_q_value + alpha * (gamma * cumulative_reward - current_q_value)

            if episode_complete:
                # print(Q_values)
                break


if __name__ == "__main__":
    # Another feasible policy
    print("Evaluating a north-only policy")
    north_policy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print_policy_value(north_policy)
    print("Improve the north-only policy")
    optimal_policy = policy_improvement(north_policy)
    print_policy_value(optimal_policy)

    # Create policy through value iteration
    print("Using value iteration to create policy")
    value_policy, value = value_iteration()
    print_policy_value(value_policy)

    # Find optimal policy via Q-learning
    print("Using Q-learning to find optimal policy")
    q_learn(10000)
    print("Q-table")
    print(Q_values)
    print_policy(np.argmax(Q_values, axis=1))

