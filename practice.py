# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import gym, threading, queue
#
# EP_MAX = 1000
# EP_LEN = 500
# N_WORKER = 4  # parallel workers
# GAMMA = 0.9  # reward discount factor
# A_LR = 0.0001  # learning rate for actor
# C_LR = 0.0001  # learning rate for critic
# MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
# UPDATE_STEP = 15  # loop update operation n-steps
# EPSILON = 0.2  # for clipping surrogate objective
# GAME = 'LunarLanderContinuous-v2'
#
# env = gym.make(GAME)
# S_DIM = env.observation_space.shape[0]
# A_DIM = env.action_space.shape[0]
# print(S_DIM,A_DIM)
#
#
# EP_TEST = 100
# file = open('data.txt', 'w')
# for _ in range(EP_TEST):
#     x = seeding.np_random(_)[0].uniform(0, 20)
#     x = (x - 600 / 30.0 / 2) / (600 / 30.0 / 2)
#     y = 0.9
#     s = env.reset()
#     s[0] = x
#     s[1] = y
#     a = ppo.choose_action(s)
#     data = np.concatenate((s, a))
#     file.write("\n"+str(data)+"\tLEFT")
# file.close()

a = "s"
print(a[1:])