
# First Q - Learning Project

from __future__ import print_function

import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6        # length of the 1-d world
EPSILON = 0.9       # greedy policy,
ALPHA = 0.1         # learning rate
LAMBDA = 0.9        # discount factor,
MAX_EPISODE = 13    # max round
FRESH_TIME = 0.1    #
ACTIONS = ['left', 'right']     # actions


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    print(table)
    return table


# build_q_table(N_STATES, ACTIONS)


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(s, a):
    if a == 'right':
        if s == N_STATES - 2:
            s_ = 'terminal'
            r = 1
        else:
            s_ = s + 1
            r = 0
    else:
        r = 0
        if s == 0:
            s_ = s
        else:
            s_ = s - 1
    return s_, r


def update_env(s, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if s == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                 ', end='')
    else:
        env_list[s] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table_i = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODE):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table_i)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table_i.ix[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table_i.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table_i.ix[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table_i


if __name__ == "__main__":

    s1 = pd.Series(
        [0, 0, 0, 0],
        index=['1', '3', '2', '0'],
        name='[5.0, 5.0, 35.0, 35.0]',
        # dtype=object,
    )
    # s1 = s1.reindex(np.random.permutation(s1.index))
    s2 = s1.idxmax()

    q_table = rl()




    print('\r\nQ-table:\n')
    print(q_table)







