import tensorflow as tf
import numpy as np


class DPPO(object):
    def __init__(
            self,
            n_actions,
            n_features,
            update_step,
            a_lr=0.0001,
            c_lr=0.0002,
            ep_max=1000,
            ep_len=200,
            gamma=0.9,
            batch_size=32,
            epsilon=0.2,
            trainable = True,
            ckpt_path = False
            ):

        self.s_dim = n_features
        self.a_dim = n_actions
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.ep_max = ep_max
        self.ep_len = ep_len
        self.update_step = update_step

        self.tfs = tf.placeholder(tf.float32, [None, n_features], 'state')

        with tf.Session() as self.sess:
            # critic
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,trainable=trainable)
            # l2 = tf.layers.dense(l1, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1,trainable=trainable)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(c_lr).minimize(self.closs)

            # actor
            pi, pi_params = self._build_anet('pi', trainable=trainable)
            oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            self.tfa = tf.placeholder(tf.float32, [None, n_actions], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
            surr = ratio * self.tfadv  # surrogate loss

            self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
                surr,
                tf.clip_by_value(ratio, 1. - epsilon, 1. + epsilon) * self.tfadv))

            self.atrain_op = tf.train.AdamOptimizer(a_lr).minimize(self.aloss)
            
            if not ckpt_path:
                self.sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.Saver()
                saver.restore(self.sess, ckpt_path)

    def update(self,COORD,GLOBAL_EP,UPDATE_EVENT,QUEUE,UPDATE_STEP,ROLLING_EVENT):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < self.ep_max:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :self.s_dim], data[:, self.s_dim: self.s_dim + self.a_dim], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            # l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
