import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    # Activation functions etc
    def forward(self, x):
        x = F.relu(self.layer_1(x)) # rectifier activation function
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))   # tanh activation function
        return x

