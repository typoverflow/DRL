import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import copy


preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((96, 80)), 
    transforms.CenterCrop(84), 
    # transforms.ToTensor(), 
])

def preprocessing(frame):
    frame = Image.fromarray(frame)
    frame = preprocess(frame)
    return np.asarray(frame, dtype='float64')/255

class Paras(object):
    learning_rate = 0.001
    batch_size = 32
    gamma = 0.99
    buffer_capacity = 10000
    niteration = 100
    agent_hist = 4

    init_epsilon = 1
    final_epsilon = 0.1
    epsilon_decay = 1/10000

    target_update_freq = 1000

    npixel = 84*84
    lpixel = 84

class Env(object):
    env = gym.make("SpaceInvaders-v0")
    frame_shape = env.observation_space.shape
    action_num = env.action_space.n

    crop_shape = (84, 84)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), stride=4)
        self.conv1.weight.data.normal_(0, 0.1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2)
        self.conv2.weight.data.normal_(0, 0.1)

        self.hidden3 = nn.Linear(81*32, 256)
        self.hidden3.weight.data.normal_(0, 0.1)
        self.hidden3.bias.data.normal_(0, 0.1)

        self.output = nn.Linear(256, Env.action_num, bias=False)
        self.output.weight.data.normal_(0, 0.1)


    def forward(self, X):
        """
            input: input of shape (Minibatch, 4, 84, 84)
            output: tensor of shape(Minibatch, action_num)
        """
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = X.view(X.size(0), -1)
        X = self.hidden3(X)
        action_values = self.output(X)
        return action_values

    
class DQN():
    def __init__(self):
        self.lr = Paras.learning_rate
        self.gamma = Paras.gamma
        
        self.epsilon = self.init_epsilon = Paras.init_epsilon
        self.final_epsilon = Paras.final_epsilon
        self.epsilon_decay = Paras.epsilon_decay

        self.buffer_cnt = 0
        self.buffer_cur = 0
        self.buffer = np.zeros((Paras.buffer_capacity, 1+1+8*Env.crop_shape[0] * Env.crop_shape[1]))
        

        self.target_net = self.eval_net = SimpleCNN()

        self.global_step = 0

        self.recent_frames = np.zeros((4, 84, 84))
        self.recent_cur = 0
        self.recent_cnt = 0

        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.1)
        self.loss_func = nn.MSELoss("mean")

    def choose_action(self, frame):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, Env.action_num)
        else:
            tmp = np.vstack([
                self.recent_frames[(self.recent_cur-3)%4][None, :], 
                self.recent_frames[(self.recent_cur-2)%4][None, :], 
                self.recent_frames[(self.recent_cur-1)%4][None, :], 
                frame[None, :]
            ])
            action_values = self.eval_net.forward(torch.FloatTensor(tmp[None, :]))[0]
            return action_values.argmax()

    def store_transition(self, frame, action, reward, next_frame):
        self.recent_frames[self.recent_cur] = frame
        self.recent_cnt += 1
        self.recent_cur = (self.recent_cur+1)%4

        transition = np.hstack([
            self.recent_frames[(self.recent_cur-4)%4].flatten(), 
            self.recent_frames[(self.recent_cur-3)%4].flatten(), 
            self.recent_frames[(self.recent_cur-2)%4].flatten(), 
            self.recent_frames[(self.recent_cur-1)%4].flatten(), 
            self.recent_frames[(self.recent_cur-3)%4].flatten(), 
            self.recent_frames[(self.recent_cur-2)%4].flatten(), 
            self.recent_frames[(self.recent_cur-1)%4].flatten(), 
            next_frame.flatten(), 
            np.asarray([action, reward])
        ])

        self.buffer[self.buffer_cur] = transition
        self.buffer_cnt += 1
        self.buffer_cur = (self.buffer_cur+1)%Paras.buffer_capacity

        return transition

    def learn(self):
        if self.global_step % Paras.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.global_step += 1
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay

        sample_index = np.random.choice(min(self.buffer_cnt, Paras.buffer_capacity), size=Paras.batch_size)
        batch = self.buffer[sample_index, :]
        batch_state = np.reshape(batch[:, :4*84*84], (-1, 4, 84, 84))
        batch_state_prime = np.reshape(batch[:, 4*84*84:2*4*84*84], (-1, 4, 84, 84))
        batch_action = batch[:, -2]
        batch_reward = batch[:, -1]

        batch_state = torch.FloatTensor(batch_state)
        batch_state_prime = torch.FloatTensor(batch_state_prime)
        batch_action = torch.LongTensor(batch_action.astype(int)[:, None])
        batch_reward = torch.FloatTensor(batch_reward[:, None])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(Paras.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def clear_recent(self):
        self.recent_cur = self.recent_cnt = 0
        self.recent_frames = np.zeros((4, 84, 84))


def main():
    try:
        dqn = DQN()
        episodes = 1000
        reward_list = []
        # plt.ion()
        print("Start training...")
        for i in range(episodes):
            frame = Env.env.reset()
            frame = preprocessing(frame)
            # plt.imshow(frame)
            # plt.show()
            total_reward = 0
            while True:
                # Env.env.render()
                action = dqn.choose_action(frame)
                next_frame, reward, done, _ = Env.env.step(action)
                # plt.imshow(frame)
                # plt.show()
                next_frame = preprocessing(next_frame)

                dqn.store_transition(frame, action, reward, next_frame)

                total_reward += reward
                if dqn.buffer_cnt >= Paras.buffer_capacity:
                    dqn.learn()
                
                if _["ale.lives"] < 3:
                    print("finish episode {}, total reward is {}.".format(i, total_reward))
                    # assert False
                    break
                
                frame = copy.deepcopy(next_frame)
                # plt.pause(0.001)
                # plt.clf()
            
            reward_list.append(total_reward)
    except:
        print(reward_list)
    
    # plt.plot(reward_list)
    # plt.show()

if __name__ == "__main__":
    main()










    


    
            






















if __name__ == "__main__": 
    pass

    # plt.imshow()