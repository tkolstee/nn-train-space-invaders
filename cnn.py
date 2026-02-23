import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class AtariCNN(nn.Module):
    def __init__(self, action_space_n):
        super(AtariCNN, self).__init__()
        # Input: 4 grayscale frames stacked (4, 84, 84)
        
        # Layer 1: 32 filters, 8x8 kernel, stride 4
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        
        # Layer 2: 64 filters, 4x4 kernel, stride 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        
        # Layer 3: 64 filters, 3x3 kernel, stride 1
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully Connected Layers
        # The 7x7x64 output from conv3 is flattened to 3136 units
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_space_n)

    def forward(self, x):
        # Normalize pixel values from [0, 255] to [0, 1]
        x = x.float() / 255.0
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for the linear layers
        x = x.reshape(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        return self.fc2(x) # Returns Q-values for each action

class ReplayBuffer:
    def __init__(self, capacity):
        # Atari typically uses a capacity of 100,000 to 1,000,000 steps
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state and next_state are the 4-frame stacks (4, 84, 84)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sampling breaks the temporal correlation between frames
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
