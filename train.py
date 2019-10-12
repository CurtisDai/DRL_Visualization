import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.utils import seeding
import tensorflow as tf
from DPPO import DPPO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




FLAGS = tf.flags.FLAGS
FLAGS.DEFINE_integer('EP_MAX', 2000, 'number of episodes in total')
tf.flags.DEFINE_integer('EP_LEN', 100, 'length of one episode')
tf.flags.DEFINE_float('N_WORKER', 0.9, 'parallel workers')
tf.flags.DEFINE_integer('GAMMA', 2, 'reward discount factor')
tf.flags.DEFINE_float('A_LR', 0.0001, 'learning rate for actor')
tf.flags.DEFINE_float('C_LR', 0.0001, 'learning rate for critic')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_integer('MIN_BATCH_SIZE', 64, 'minimum batch size for updating PPO')
tf.flags.DEFINE_integer('UPDATE_STEP', 15, 'loop update operation n-steps')
tf.flags.DEFINE_float('EPSILON', 0.2, 'for clipping surrogate objective')
tf.flags.DEFINE_string('GAME', 'LunarLanderContinuous-v2', 'the gym game you wanna play')


env = gym.make(FLAGS.GAME)
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]


if __name__ == '__main__':
    GLOBAL_PPO = DPPO(A_DIM,
                      S_DIM,
            FLAGS.UPDATE_STEP,
            a_lr=FLAGS.A_LR,
            c_lr=FLAGS.C_LR,
            ep_max=FLAGS.EP_MAX,
            ep_len=FLAGS.EP_LEN,
            gamma=FLAGS.GAMMA,
            batch_size=32,
            epsilon=0.2,
            trainable = True,
            ckpt_path = False
            )
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.ion()
    plt.show()


    file = open('data2.txt', 'w')
    for _ in range(EPOCH_TEST):
        x = seeding.np_random(_)[0].uniform(0, 20)
        x = (x - 600 / 30.0 / 2) / (600 / 30.0 / 2)
        y = 0.9
        s = env.reset()
        s[0] = x
        s[1] = y
        a = GLOBAL_PPO.choose_action(s)
        data = np.concatenate((s, a))
        string = ""
        for i in data:
            string = string + str(i) + " "
        file.write("\n" + string + "\tRIGHT")
    file.close()

    saver = tf.train.Saver() # save the model
    saver.save(GLOBAL_PPO.sess, "./model/model.ckpt")


    env = gym.make(GAME)
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]