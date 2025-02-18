'''
************************************Copyright (c)******************************************************
**                      		Shannxi Normal University
**                               http://www.snnu.edu.cn/
**--------------File Info------------------------------------------------------------------------------
** File name:           AM.py
** Created by:          zhaochenyang
** Created date:        2024-10-20
** Version:             V1.0
** Descriptions:         Adaptive mixing
********************************************************************************************************
'''

import numpy as np
import random
import time
from numba import jit
import matplotlib.pyplot as plt


# Reward Matrix
@jit(nopython=True)
def PayoffFunction(A1, A2, x):
    if A1 != A2:
        if A1 == 1:  # CD
            off = -x
        else:
            off = 1 + x
    else:
        if A1 == 1:  # CC
            off = 1
        else:
            off = 0
    return off


@jit(nopython=True)
def PD_Q_game(N, N_pin, epsilon, gamma, alpha, Max_t, L, x, thresholds):
    percent = np.zeros(Max_t)
    pc_c = np.zeros(Max_t)
    qc_c = np.zeros(Max_t)
    Q_table = np.random.randn(N, 6, 2)
    Action = np.random.randint(0, 2, N)
    State = np.zeros(N)
    Reward = np.zeros(N)
    State_new = np.zeros(N)
    di = [1, -1, L, -L]
    N_num = np.arange(0, N)
    pining_agent = np.random.choice(10000, N_pin, replace=False)  # Select HCC
    Q_agent = np.delete(N_num, pining_agent)
    Reward_old = np.zeros(N)
    N_Q = np.zeros(Max_t)
    N_P = np.zeros(Max_t)

    # initialization state
    for i in range(N):
        itC_num = 0
        for j in range(K):
            it = ((i + di[j]) % N + N) % N  # periodic boundary condition
            itC_num += Action[it]
        State[i] = Action[i] + itC_num

    for t in range(0, Max_t):
        pc_num = 0
        qc_num = 0
        pining = np.array([])
        Q = np.array([])
        N_P[t] = len(pining_agent)
        N_Q[t] = len(Q_agent)

        for i in pining_agent:
            if (State[i] - Action[i]) >= thresholds:
                Action[i] = 1
                pc_num += 1
            else:
                Action[i] = 0

        for i in Q_agent:
            p = random.random()
            if p < epsilon:
                Action[i] = random.randint(0, 1)
            else:
                Action[i] = Q_table[i][int(State[i]), :].argmax()
            if Action[i] == 1:
                qc_num += 1

        C_num = np.sum(Action)
        for i in range(0, N):
            Payoff = 0
            itC_num = 0
            for j in range(0, K):
                it = ((i + di[j]) % N + N) % N
                itC_num += Action[it]
                A1 = int(Action[i])
                A2 = int(Action[it])
                off = PayoffFunction(A1, A2, x)
                Payoff = Payoff + off
            Reward[i] = Payoff / 4
            State_new[i] = Action[i] + itC_num

        # Update Strategy
        for i in range(N):
            if Reward[i] < Reward_old[i]:
                if i in Q_agent:
                    pining = np.append(pining, i)
                if i in pining_agent:
                    Q = np.append(Q, i)
            else:
                if i in Q_agent:
                    Q = np.append(Q, i)
                if i in pining_agent:
                    pining = np.append(pining, i)
        pining_agent = np.copy(pining)
        Q_agent = np.copy(Q)

        for i in range(N):
            Qmax_new = Q_table[i][int(State_new[i]), :].max()
            Q_table[i][int(State[i])][int(Action[i])] = (1 - alpha) * Q_table[i][int(State[i])][
                int(Action[i])] + alpha * (Reward[i] + gamma * Qmax_new)

        State = np.copy(State_new)
        percent[t] = C_num / N
        pc_c[t] = pc_num / len(pining_agent)
        qc_c[t] = qc_num / len(Q_agent)

    return percent, pc_c, qc_c


if __name__ == '__main__':
    np.random.seed(2024)                                                  # Random seed
    start_time = time.time()                                              # Start time
    x = 0.1                                                               # Game parameters
    N = 10000                                                             # Size of population
    L = 100                                                               # System size
    Max_t = 8000000                                                       # Evolutionary time
    epsilon = 0.01                                                        # E-greedy
    K = 4                                                                 # Number of neighbors
    alpha = 0.1                                                           # Learning rate
    gamma = 0.9                                                           # Discount factory
    rho = 0.3                                                             # The density of HCC
    thresholds = 3                                                        # Expected threshold
    N_pin = int(N * rho)
    fc, pc, qc = PD_Q_game(N, N_pin, epsilon, gamma, alpha, Max_t, L, x, thresholds)
    end_time = time.time()
    print(f"spend time {end_time - start_time}")                          # End time

    # Save data

    t_log = int(pow(10, 3))
    t_log_exp = 3
    f = open('AM_TH3_rho=0.3.txt', 'w')
    for t in range(Max_t):
        if t <= 10:
            print(t, fc[t], pc[t], file=f)
        if t > 10 and t < 1000 and t % 10 == 0:
            print(t, fc[t], pc[t], qc[t], file=f)
        else:
            if t == t_log:
                print(t, fc[t], pc[t], qc[t], file=f)
                t_log_exp += 0.0025
                t_log = int(pow(10, t_log_exp))
    f.close()