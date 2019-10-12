import gym
import numpy as np
class Worker(object):
    def __init__(self, wid, game, model,ep_len,min_batch_size,gamma,ep_max):
        self.wid = wid
        self.env = gym.make(game).unwrapped
        self.ppo = model
        self.ep_len = ep_len
        self.ep_max = ep_max
        self.min_batch_size = min_batch_size
        self.gamma =gamma

    def work(self,COORD,ROLLING_EVENT,QUEUE,UPDATE_EVENT):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(self.ep_len):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == self.ep_len - 1 or GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= self.ep_max:  # stop training
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP / self.ep_max * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )
