# Basic network building blocks for the different learning algorithms

import collections

import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
from torch.autograd import Variable

##if torch.cuda.is_available():
##    device = torch.device("cuda")
##    torch.set_default_tensor_type('torch.cuda.FloatTensor')
##else:
##    device = torch.device("cpu")

device = torch.device("cpu")


class DNN(nn.Module):

    def __init__(self, in_size, out_size, hidden=16):

        super(DNN, self).__init__()

        self.out_size = out_size
        if not type(out_size) == int:
            out_size = np.prod(out_size)

        # self.line = nn.Linear(in_size, out_size, bias=True)

        self.line1 = nn.Linear(in_size, hidden, bias=True)
        # self.line2 = nn.Linear(hidden, hidden, bias=True)
        self.line3 = nn.Linear(hidden, out_size, bias=True)

    def forward(self, x):

        x = x.view(x.size(0), -1).float()

        # x = self.line(x)

        x = F.relu(self.line1(x))
        # x = F.relu(self.line2(x))
        x = self.line3(x)

        if type(self.out_size) == int:
            x = x.view(x.size(0), self.out_size)
        else:
            x = x.view((x.size(0),) + self.out_size)

        return x


class PolicyDNN(nn.Module):

    def __init__(self, in_size, action_size, hidden=16):
        super(PolicyDNN, self).__init__()

        # self.line = nn.Linear(in_size, action_size, bias=True)

        self.line1 = nn.Linear(in_size, hidden, bias=True)
        # self.line2 = nn.Linear(hidden, hidden, bias=True)
        self.line3 = nn.Linear(hidden, action_size, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # x = self.line(x)
        x = F.relu(self.line1(x))
        # x = F.relu(self.line2(x))
        x = self.line3(x)

        x = F.softmax(x, dim=-1)
        # x = Categorical(F.softmax(x, dim=-1))

        return x


##################################################

class CNN(nn.Module):

    def __init__(self, board_size, channels, out_size, convs=8, hidden=16):

        # assumes input of shape (board_size x board_size x channels)

        super(CNN, self).__init__()

        self.out_size = out_size
        if not type(out_size) == int:
            out_size = np.prod(out_size)

        self.conv1 = nn.Conv2d(channels, convs, kernel_size=5, stride=1)
        self.line1 = nn.Linear(convs * (board_size - 4) ** 2, hidden, bias=True)
        self.line2 = nn.Linear(hidden, out_size, bias=True)

        self.board_size = board_size

    def forward(self, x):

        x = x.view(x.size(0), 3, self.board_size, self.board_size)
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.line1(x))
        x = self.line2(x)
        if type(self.out_size) == int:
            x = x.view(x.size(0), self.out_size)
        else:
            x = x.view((x.size(0),) + self.out_size)

        return x


class PolicyCNN(nn.Module):

    def __init__(self, board_size, channels, action_size, convs=8, hidden=16):
        # assumes input of shape (board_side x board_side x channels)

        super(PolicyCNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, convs, kernel_size=5, stride=1)
        self.line1 = nn.Linear(convs * (board_size - 4) ** 2, hidden, bias=True)
        self.line2 = nn.Linear(hidden, action_size, bias=True)

        self.board_size = board_size

    def forward(self, x):
        x = x.view(x.size(0), 3, self.board_size, self.board_size)
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.line1(x))
        x = x.view(x.size(0), -1)
        x = self.line2(x)
        x = F.softmax(x, dim=-1)

        return x


##################################################

# these networks take as input a board, as well as one additional parameter
# (needed for certain agent types)

class CNN2(nn.Module):

    def __init__(self, board_size, channels, out_size, convs=8, hidden=16):

        # assumes input of shape (board_side x board_side x channels)

        super(CNN2, self).__init__()

        self.out_size = out_size
        if not type(out_size) == int:
            out_size = np.prod(out_size)

        self.conv1 = nn.Conv2d(channels, convs, kernel_size=5, stride=1)
        self.line1 = nn.Linear(convs * (board_size - 4) ** 2 + 1, hidden, bias=True)
        self.line2 = nn.Linear(hidden, out_size, bias=True)

        self.board_size = board_size

    def forward(self, x):

        x, y = torch.split(x, x.size(1) - 1, dim=1)
        x = x.view(x.size(0), 3, self.board_size, self.board_size)

        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.line1(x))
        x = self.line2(x)
        if type(self.out_size) == int:
            x = x.view(x.size(0), self.out_size)
        else:
            x = x.view((x.size(0),) + self.out_size)

        return x


class PolicyCNN2(nn.Module):

    def __init__(self, board_size, channels, action_size, convs=8, hidden=16):
        # assumes input of shape (board_side x board_side x channels)

        super(PolicyCNN2, self).__init__()

        self.conv1 = nn.Conv2d(channels, convs, kernel_size=5, stride=1)
        self.line1 = nn.Linear(convs * (board_size - 4) ** 2 + 1, hidden, bias=True)
        self.line2 = nn.Linear(hidden, action_size, bias=True)

        self.board_size = board_size

    def forward(self, x):
        x, y = torch.split(x, x.size(1) - 1, dim=1)
        x = x.view(x.size(0), 3, self.board_size, self.board_size)

        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.line1(x))
        x = x.view(x.size(0), -1)
        x = self.line2(x)
        x = F.softmax(x, dim=-1)

        return x


##################################################

class ContinuousPolicyDNN(nn.Module):

    def __init__(self, in_size, action_size, hidden=16):
        super(ContinuousPolicyDNN, self).__init__()

        self.line1 = nn.Linear(in_size, hidden, bias=True)
        # self.line2 = nn.Linear(hidden, hidden, bias=True)
        self.mu = nn.Linear(hidden, action_size, bias=True)
        self.var = nn.Linear(hidden, action_size, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.line1(x))
        # x = F.relu(self.line2(x))
        mu = torch.tanh(self.mu(x))
        var = F.softplus(self.var(x)) + 1e-5

        return mu, var


class ContinuousPolicyCNN(nn.Module):

    def __init__(self, board_size, channels, action_size, convs=8, hidden=16):
        # assumes input of shape (board_side x board_side x channels)

        super(ContinuousPolicyCNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, convs, kernel_size=5, stride=1)
        self.line1 = nn.Linear(convs * (board_size - 4) ** 2, hidden, bias=True)
        self.mu = nn.Linear(hidden, action_size, bias=True)
        self.var = nn.Linear(hidden, action_size, bias=True)

        self.board_size = board_size

    def forward(self, x):
        x = x.view(x.size(0), 3, self.board_size, self.board_size)
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.line1(x))
        x = x.view(x.size(0), -1)
        mu = torch.tanh(self.mu(x))
        var = F.softplus(self.var(x)) + 1e-5

        return mu, var


##################################################

# these networks take as input a board, as well as one additional parameter
# (needed for certain agent types)

class ContinuousPolicyCNN2(nn.Module):

    def __init__(self, board_size, channels, action_size, convs=8, hidden=16):
        # assumes input of shape (board_side x board_side x channels)

        super(ContinuousPolicyCNN2, self).__init__()

        self.conv1 = nn.Conv2d(channels, convs, kernel_size=5, stride=1)
        self.line1 = nn.Linear(convs * (board_size - 4) ** 2 + 1, hidden, bias=True)
        self.mu = nn.Linear(hidden, action_size, bias=True)
        self.var = nn.Linear(hidden, action_size, bias=True)

        self.board_size = board_size

    def forward(self, x):
        x, y = torch.split(x, x.size(1) - 1, dim=1)
        x = x.view(x.size(0), 3, self.board_size, self.board_size)

        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.line1(x))
        x = x.view(x.size(0), -1)
        mu = torch.tanh(self.mu(x))
        var = F.softplus(self.var(x)) + 1e-5

        return mu, var


##################################################


def make_network(network_purpose, network_type, in_size, hidden, out_size,
                 is_action_cont=False, extra_input=False, ):
    assert network_purpose in ['policy', 'prediction']
    assert network_type in ['DNN', 'CNN']

    if network_purpose == 'prediction':

        if network_type == 'DNN':
            if extra_input:
                net = DNN(in_size + 1, out_size, hidden)
            else:
                net = DNN(in_size, out_size, hidden)

        if network_type == 'CNN':
            if extra_input:
                net = CNN2(int((in_size / 3) ** 0.5), channels=3, out_size=out_size, convs=hidden, hidden=hidden)
            else:
                net = CNN(int((in_size / 3) ** 0.5), channels=3, out_size=out_size, convs=hidden, hidden=hidden)

    if network_purpose == 'policy':

        if network_type == 'DNN':
            if is_action_cont:
                if extra_input:
                    net = ContinuousPolicyDNN(in_size + 1, out_size, hidden)
                else:
                    net = ContinuousPolicyDNN(in_size, out_size, hidden)
            else:
                if extra_input:
                    net = PolicyDNN(in_size + 1, out_size, hidden)
                else:
                    net = PolicyDNN(in_size, out_size, hidden)

        if network_type == 'CNN':
            raise Exception("Untested")
            # if is_action_cont:
            #     if extra_input:
            #         net = ContinuousPolicyCNN2(int((in_size / 3) ** 0.5), channels=3,
            #                                    convs=hidden, action_size=out_size, hidden=hidden)
            #     else:
            #         net = ContinuousPolicyCNN(int((in_size / 3) ** 0.5), channels=3,
            #                                   convs=hidden, action_size=out_size, hidden=hidden)
            # else:
            #     if extra_input:
            #         net = PolicyCNN2(int((in_size / 3) ** 0.5), channels=3,
            #                          convs=hidden, action_size=out_size, hidden=hidden)
            #     else:
            #         net = PolicyCNN(int((in_size / 3) ** 0.5), channels=3,
            #                         convs=hidden, action_size=out_size, hidden=hidden)

    return net


##################################################

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.memory = collections.deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = collections.namedtuple("Experience",
                                                  field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])

    def add(self, state, action, reward, next_state, done):

        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, sample_all=False):

        if sample_all:
            experiences = self.memory
        else:
            experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state.cpu() for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # actions = torch.from_numpy(np.vstack([int(e.action) for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state.cpu() for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


##################################################

class LagrangeLambda(nn.Module):

    def __init__(self, param_sizes):
        super(LagrangeLambda, self).__init__()

        self.lambdas = nn.ParameterList(
            [nn.Parameter(torch.zeros(p, requires_grad=True, dtype=torch.float)) for p in param_sizes])
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, grads):
        flat_lambdas = torch.cat([torch.flatten(l) for l in self.lambdas])
        return torch.sum(flat_lambdas * grads, dim=1)
