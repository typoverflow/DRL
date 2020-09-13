import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import copy
from torch.utils.tensorboard import SummaryWriter

class BaseAgent(object):
    def __init__(self):
        self.global_step = 0
        self.global_epoch = 0
        self.reward_list = []
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.writer = SummaryWriter(log_dir="./log")

    def policy(self, state):
        raise NotImplementedError

    def ep_policy(self, state):
        raise NotImplementedError

    def train(self, env, epochs):
        raise NotImplementedError

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))

    def eval(self, env, epochs=1, is_render=False):
        env = copy.deepcopy(env)
        total_reward = 0
        for _ in range(epochs):
            s = env.reset()
            while True:
                if is_render:
                    self.env.render()
                a = self.policy(s)
                s_, r, done, _ = env.step(a)
                total_reward += r
                if done:
                    break
                s = s_
        total_reward /= epochs
        self.reward_list.append(total_reward)

        self.writer.add_scalar('Reward/eval', total_reward, self.global_epoch)
        return total_reward

    def plot_reward(self, title, is_save=True, is_show=True):
        plt.plot(self.reward_list)
        plt.title(title)
        plt.xlabel("global step")
        plt.ylabel("reward")
        if is_save:
            plt.savefig(title+".png")
        if is_show:
            plt.show()


class RandomAgent(BaseAgent):
    def __init__(self, N_action, eval_interval):
        super(RandomAgent, self).__init__()
        self.N_action = N_action
        self.eval_interval = eval_interval

    def policy(self, state):
        return np.random.randint(0, self.N_action)

    def ep_policy(self, state):
        return np.random.randint(0, self.N_action)

    def train(self, env, epochs):
        env = copy.deepcopy(env)
        for _ in range(epochs):
            s = env.reset()
            while True:
                a  = self.ep_policy(s)
                s_, r, done, _ = env.step(a)

                self.global_step += 1

                if done:
                    break
                s = s_
            self.global_epoch += 1

            if self.global_epoch % self.eval_interval == 0:
                eval_r = self.eval(epochs=5, is_render=False)
                self.writer.add_scalar("Reward/Random_Agent", eval_r, self.global_epoch)
                print("Epoch: {},\tReward: {}.".format(self.global_epoch, eval_r))

        
    

        
                