import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from threading import active_count
import gym
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy
from tools.base_agent import BaseAgent
from torch.distributions import Categorical
import copy
import itertools

class Paras(object):
    learning_rate = 0.001
    N_hidden = 256
    N_rollout = 32

class Env(object):
    env = gym.make("CartPole-v0")
    N_input = env.observation_space.shape[0]
    N_output = env.action_space.n

class Simple_Actor(nn.Module):
    def __init__(self, N_input, N_output, N_hidden):
        super(Simple_Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(N_input, N_hidden), 
            nn.ReLU(), 
            nn.Linear(N_hidden, N_output), 
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        probs = self.actor(input)
        return Categorical(probs)


class Simple_Critic(nn.Module):
    def __init__(self, N_input, N_hidden):
        super(Simple_Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(N_input, N_hidden), 
            nn.ReLU(), 
            nn.Linear(N_hidden, 1)
        )

    def forward(self, input):
        return self.critic(input)

    

class A2CAgent(BaseAgent):
    def __init__(self, is_cnn, is_recurrent, use_cuda, **kargs):
        # super(BaseAgent, self).__init__()
        super().__init__()
        self.is_cnn = is_cnn
        self.is_recurrent = is_recurrent
        
        if not (is_cnn or is_recurrent):
            for arg in ["N_input", "N_hidden", "N_output"]:
                assert arg in kargs.keys()
            self.actor = Simple_Actor(kargs["N_input"], kargs["N_output"], kargs["N_hidden"])
            self.critic = Simple_Critic(kargs["N_input"], kargs["N_hidden"])
        else:
            raise NotImplementedError

        if use_cuda:
            if not torch.cuda.is_available():
                assert False
            self.actor = self.actor
            self.critic = self.critic

        self.gamma = kargs["gamma"]

        self.actor_optim = optim.Adam(self.actor.parameters())
        self.critic_optim = optim.Adam(self.critic.parameters())

    def policy(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).probs.argmax().item()
    
    def train(self, env, epochs):
        env = copy.deepcopy(env)
        for ep in range(epochs):
            state = torch.FloatTensor(env.reset()).unsqueeze(0)
            log_probs = []
            rewards = []
            values = []
            masks = []
            entropy = 0
            done = False

            for t in range(Paras.N_rollout):
                dist = self.actor(state)
                value = self.critic(state)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                next_state, reward, done, _ = preprocess(env.step(action.item()))

                entropy += dist.entropy().mean()

                mask = 1-done

                log_probs.append(log_prob)
                rewards.append(reward)
                masks.append(mask)
                values.append(value[0])

                state = next_state

                if done: break
            
            returns = []
            if done:
                R = 0
            else:
                R = self.critic(state).item()
            for t in reversed(range(len(rewards))):
                R = rewards[t] + self.gamma * R * masks[t]
                returns.insert(0, torch.FloatTensor([R]))

            log_probs = torch.cat(log_probs, dim=0)
            values = torch.cat(values, dim=0)
            returns = torch.cat(returns, dim=0)
            advantage = returns-values

            actor_loss = -(log_probs*advantage.detach()).mean() - 0.01*entropy
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss+critic_loss
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

            if ep % 20 == 0:
                print("Epoch {}\tReward {}.".format(ep, self.eval(env)))
            if ep % 1000 == 0 and (not ep == 0):
                self.plot_reward("CartPole", True, True)

def preprocess(t):
    state, reward, done, _ = t
    state = torch.FloatTensor(state).unsqueeze(0)
    # reward = torch.FloatTensor(reward).unsqueeze(0)
    return state, reward, done, _

if __name__ == "__main__":
    agent = A2CAgent(False, False, True, N_input=Env.N_input, 
                                          N_output=Env.N_output, 
                                        N_hidden=Paras.N_hidden, 
                                          gamma=0.99)
    agent.train(Env.env, 10000)
