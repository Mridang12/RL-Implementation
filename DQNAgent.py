from io import TextIOBase
import numpy as np
from collections import deque, namedtuple
from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import torch.optim as optim

# Setup Experience Replay
Experience = namedtuple('Experience', ['state', 'action',
                                        'n_state', 'reward', 'done'])

class ExperienceReplay():
    def __init__(self, capacity):
        self.mem = deque([], maxlen=capacity)

    def push(self, experience):
        self.mem.append(experience)

    def sample(self, batch_size):
        sample = random.sample(self.mem, batch_size)
        states = [x.state for x in sample] 
        actions = [x.action for x in sample]  
        n_states = [x.n_state for x in sample]  
        rewards = [x.reward for x in sample]
        dones = [x.done for x in sample]

        return torch.stack(states), torch.stack(actions).unsqueeze(1), torch.stack(n_states), torch.stack(rewards).unsqueeze(1), torch.stack(dones).unsqueeze(1) 


    def __len__(self):
        return len(self.mem)


#Neural Network
class Net(nn.Module):

    # layerSizes[0] should be equal to len of the state vector
    # layerSizes[-1] should be equal to len of the action space (2 for flappy) 
    def __init__(self, layerSizes = []):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(0, len(layerSizes) - 1):
            self.layers.append(nn.Linear(layerSizes[i], layerSizes[i + 1]))
        pass

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x
#DQN Agent
class DQNAgent():

    def __init__(self,layerSizes, epsilon, eps_decay, min_eps, mem_size, batch_size, discount_fact, update_freq, target_update_freq, lr, num_actions):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.discount_fact = discount_fact
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.lr = lr
        self.layerSizes = layerSizes
        self.epsilon = epsilon
        self.action_space = num_actions
        self.decay_rate = eps_decay
        self.min_eps = min_eps

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policyNet = Net(self.layerSizes).to(self.device)
        self.targetNet = Net(self.layerSizes).to(self.device)
        
        #Using optim.Adam might be a better idea but Gradient Descent works okay too
        #GD Might need more episodes to convert
        self.optimizer = optim.SGD(self.policyNet.parameters(), lr = self.lr)
        
        self.replay = ExperienceReplay(self.mem_size)
        self.t_step = 0
        self.target_step = 0

        #debug
        self.loss_val = 0
        pass
    
    def getAction(self, state):
        state = torch.tensor(state, device=self.device).float().unsqueeze(0)
        with torch.no_grad():
            Q_vals = self.policyNet(state)
        
        if random.random() > self.epsilon:
            return np.argmax(Q_vals.cpu().data.numpy())
        else:
            return random.randint(0, self.action_space - 1)
        

    def step(self, state, action, reward, next_state, done):
        self.replay.push(Experience(tensor(state, device=self.device).float(), 
                                    tensor(action, device = self.device, dtype=torch.int64), 
                                    tensor(next_state, device=self.device).float(), 
                                    tensor(reward, device=self.device).float(),
                                    tensor(done, device = self.device)))

        self.t_step += 1
        if self.t_step % self.update_freq == 0:
            self.t_step = 0
            if len(self.replay) > self.batch_size:
                if self.epsilon > 0:
                    self.epsilon = max(self.epsilon * self.decay_rate, self.min_eps)
                self.train()

        self.target_step += 1
        if self.target_step % self.target_update_freq == 0:
            self.target_step = 0
            self.targetNet.load_state_dict(self.policyNet.state_dict())

    def train(self):
        states, actions, next_states, rewards, dones = self.replay.sample(self.batch_size)
        criterion = nn.SmoothL1Loss()

        predictions = self.policyNet(states).gather(1, actions)

        with torch.no_grad():
            target_Qvals = self.targetNet(next_states).detach().max(1).values.unsqueeze(1)

        targets = rewards + self.discount_fact*target_Qvals*(1-dones)
        loss = criterion(predictions, targets).to(self.device)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_val += loss.item()

        pass

