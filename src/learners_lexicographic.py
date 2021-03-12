# Our (lexicographic) learning algorithms

from networks import *

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
from torch.autograd import Variable

# Constants
UPDATE_EVERY = 4
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 32
EPSILON = 0.05
LAMBDA_RL_2 = 0.05
UPDATE_EVERY_EPS = 32
SLACK = 0.04
TOL = 1
CONVERGENCE_LENGTH = 1000
CONVERGENCE_DEVIATION = 0.04
TOL2 = 1

NO_CUDA = True

class LexDQN:
    
    # lexicographic DQN
    
    def __init__(self, in_size, action_size, reward_size=2, network='DNN', hidden=16):
            
        self.t = 0                                   # total number of frames observed
        self.discount = 0.99                         # discount
        
        self.actions = list(range(action_size))
        self.reward_size = reward_size
        self.action_size = action_size
        
        if network=='DNN':
            self.model  = DNN(in_size, (reward_size, action_size), hidden)
        elif network=='CNN':
            self.model = CNN(int((in_size/3)**0.5), channels=3, 
                             out_size=(reward_size, action_size), 
                             convs=hidden, hidden=hidden)
        else:
            print('invalid network specification')
            assert False
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.optimizer = optim.Adam(self.model.parameters())
        
        if (torch.cuda.is_available() and not NO_CUDA):
            self.model.cuda()
    
    
    def act(self, state):

        if np.random.choice([True,False], p=[EPSILON, 1-EPSILON]):
            return random.choice(self.actions)
            
        Q = self.model(state)[0]
        permissible_actions = self.permissible_actions(Q)
        return random.choice(permissible_actions)
 

    def permissible_actions(self, Q):
        
        permissible_actions = self.actions
        
        for i in range(self.reward_size):
            Qi = Q[i,:]
            m = max([Qi[a] for a in permissible_actions])
            r = SLACK
            permissible_actions = [a for a in permissible_actions if Qi[a] >= m-r*abs(m)]
            
        return permissible_actions

    
    def lexmax(self, Q):
        
        a = self.permissible_actions(Q)[0]
        return Q[:,a]
    
    
    def step(self, state, action, reward, next_state, done):
        
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)
        
        if self.t % UPDATE_EVERY == 0 and len(self.memory) > BATCH_SIZE:
            experience = self.memory.sample()
            self.update(experience)
                
            
    def update(self, experiences):
 
        states, actions, rewards, next_states, dones = experiences
    
        criterion = torch.nn.MSELoss()
        self.model.train()
        
        # probably slightly suboptimal
        idx = torch.cat((actions,actions),1).reshape(-1, self.reward_size, 1)
        predictions = self.model(states.to(device)).gather(2, idx).squeeze()
            
        with torch.no_grad():
            predictions_next = self.model(next_states).detach()
            next_values = torch.stack([self.lexmax(Q) for Q in torch.unbind(predictions_next, dim=0)],dim=0)
            
        targets = rewards + (self.discount * next_values * (1-dones))
        
        loss = criterion(predictions, targets).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.model.eval()

    def save_model(self, root):
        torch.save(self.model.state_dict(), '{}-model.pt'.format(root))

    def load_model(self, root):
        self.model.load_state_dict('{}-model.pt'.format(root))

##################################################

class LexActorCritic:
    
    def __init__(self, in_size, action_size, mode, reward_size=2, second_order=False, sequential=False, network='DNN', hidden=16, continuous=False, extra_input=False):

        if mode != 'a2c' and mode != 'ppo':
            print("Error: mode must be \'a2c\' or \'ppo\'")
            return
        self.mode = mode
            
        self.t = 0                                   # total number of frames observed
        self.discount = 0.99                         # discount
        self.reward_size = reward_size

        self.action_size = action_size
        self.continuous = continuous
        
        self.actor = make_network('policy', network, in_size, hidden, action_size, continuous, extra_input)
        self.critic = make_network('prediction', network, in_size, hidden, reward_size, continuous, extra_input)
        self.mu = [0.0 for _ in range(reward_size - 1)]
        self.j = [0.0 for _ in range(reward_size - 1)]
        self.recent_losses = [collections.deque(maxlen=25) for i in range(reward_size)]

        # If updating the actor sequentially (one objective at a time) we need to keep track of which objective we're using
        self.i = 0 if sequential else None

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        # If using Adam then only the eta LR and the (relative) beta LRs matter
        self.beta = list(reversed(range(1, reward_size + 1)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, reward_size + 1)))]

        # If using second order terms
        self.second_order = second_order
        if self.second_order:
            actor_params = filter(lambda p: p.requires_grad, self.actor.parameters())
            self.lamb = [LagrangeLambda(tuple([p.shape for p in actor_params])) for _ in range(reward_size - 1)]
            self.lagrange_optimizer = [optim.Adam(ll.parameters()) for ll in self.lamb]
            self.grad_j = [0.0 for _ in range(reward_size - 1)]
            self.recent_grads = [collections.deque(maxlen=25) for i in range(reward_size - 1)]
        
        if (torch.cuda.is_available() and not NO_CUDA):
            self.actor.cuda()
            self.critic.cuda()
            if self.second_order:
                for ll in self.lamb:
                    ll.cuda()

        # If using PPO obective we need an extra hyperparemeter for weighting the loss from the KL penalty
        if mode == 'ppo':
            self.kl_weight = 1.0
            self.kl_target = 0.025
        
    
    def act(self, state):
        if self.continuous:
            mu, var = self.actor(state)
            mu = mu.data.cpu().numpy()
            sigma = torch.sqrt(var).data.cpu().numpy() 
            action = np.random.normal(mu, sigma)
            return torch.tensor(np.clip(action, -1, 1))
        else:
            return Categorical(self.actor(state)).sample()
        
    
    def step(self, state, action, reward, next_state, done):

        if self.t == 0:
            self.start_state = state
        
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)

        if self.t % BATCH_SIZE == 0:
            self.update(self.memory.sample(sample_all=True))
            self.memory.memory.clear()

            
    def update_actor(self, experiences):
    
        self.actor.train()

        loss = self.compute_loss(experiences, self.i + 1) if self.i != None else self.compute_loss(experiences, self.reward_size)
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.actor.eval()

        
    def update_critic(self, experiences):
        
        states, _, rewards, next_states, dones = experiences
    
        self.critic.train()
        
        prediction = self.critic(states.to(device))
        with torch.no_grad():
            target = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
            
        loss = nn.MSELoss()(prediction, target).to(device)
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        self.critic.eval()

    
    def update_lagrange(self):

        # Save relevant loss information for updating Lagrange parameters
        if self.i != None:
            if not self.converged():
                if self.i != self.reward_size - 1:
                    self.j[self.i] = -torch.tensor(self.recent_losses[self.i]).mean()
                    if self.second_order:
                        self.grad_j[self.i] = -torch.stack(tuple(self.recent_grads[self.i]), dim=0).mean(dim=0)
            else:
                # print("Converged for reward function {}!".format(self.i))
                self.i = 0 if self.i == self.reward_size - 1 else self.i + 1
                
        else:
            for i in range(self.reward_size - 1):
                self.j[i] = -torch.tensor(self.recent_losses[i]).mean()
                if self.second_order:
                    self.grad_j[i] = -torch.stack(tuple(self.recent_grads[i]), dim=0).mean(dim=0)

        # Update Lagrange parameters
        r = self.i if self.i != None else self.reward_size - 1
        for i in range(r):
            self.mu[i] += self.eta[i] * (self.j[i] - self.recent_losses[i][-1])
            self.mu[i] = max(self.mu[i], 0.0)
            if self.second_order:
                self.lamb[i].train()
                loss = self.lamb[i](torch.unsqueeze(self.grad_j[i] - self.recent_grads[i][-1], 0).to(device)).to(device)
                self.lagrange_optimizer[i].zero_grad()
                loss.backward()
                self.lagrange_optimizer[i].step()
                self.lamb[i].eval()


    def update(self, experiences):
        self.update_actor(experiences)
        self.update_critic(experiences)
        self.update_lagrange()

    
    def converged(self, tolerance=0.1, bound=0.01, minimum_updates=5):

        # If not enough updates have been performed, assume not converged
        if len(self.recent_losses[self.i]) < minimum_updates:
            return False
        else:
            l_mean = torch.tensor(self.recent_losses[self.i]).mean().float()
            # If the mean loss is lower than some absolute bound, assume converged
            if l_mean < bound:
                return True
            # Else check if the max of the recent losses are sufficiently close to the mean, if so then assume converged
            else:
                l_max = max(self.recent_losses[self.i]).float()
                if l_max > (1.0 + tolerance) * l_mean:
                    return False
        
        return True


    def get_log_probs(self, states, actions):

        if self.continuous:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-3))
            p2 = - torch.log(torch.sqrt(2 * math.pi * variances))
            log_probs = p1 + p2
        else:
            dists = self.actor(states.to(device))
            log_probs = torch.log(torch.gather(dists, 1, actions))

        return log_probs


    def compute_loss(self, experiences, reward_range):

        states, actions, rewards, next_states, dones = experiences
        
        # Compute weights for different components of the first order objective terms
        first_order = []
        for i in range(reward_range - 1):
            if self.i != None:
                w = self.beta[reward_range - 1] * self.mu[i]
            else:
                w = self.beta[i] + self.mu[i] * sum([self.beta[j] for j in range(i+1,reward_range)])
            first_order.append(w)
        first_order.append(self.beta[reward_range - 1])
        first_order_weights = torch.tensor(first_order)

        # If needed, compute the weights for the second order objective terms as well
        if self.second_order:
            if self.i != None:
                second_order = [self.beta[self.i] for _ in range(reward_range - 1)]
            else:
                second_order = [sum([self.beta[j] for j in range(i+1,reward_range)]) for i in range(reward_range - 1)]
                second_order.append(self.beta[reward_range - 1])
            second_order_weights = torch.tensor(second_order)

        # Computer A2C loss
        if self.mode == 'a2c':
            with torch.no_grad():
                baseline = self.critic(states.to(device))
                outcome  = rewards + (self.discount * self.critic(next_states.to(device))[:,0:reward_range] * (1-dones))
                advantage = (outcome - baseline).detach()
            log_probs = self.get_log_probs(states, actions)
            first_order_weighted_advantages = torch.sum(first_order_weights * advantage[:,0:reward_range], dim=1).to(device)
            # This will be slow in Pytorch because of the lack of forward-mode differentiation
            if self.second_order:
                scores = torch.stack(tuple([torch.cat([torch.flatten(g) for g in torch.autograd.grad(l_p.to(device), self.actor.parameters(), retain_graph=True)]) for l_p in log_probs]), dim=0).to(device)
                second_order_weighted_advantages = torch.sum(second_order_weights * advantage[:,0:reward_range], dim=1).to(device)
                second_order_terms = [self.lamb[i](scores.to(device)) * second_order_weighted_advantages for i in range(reward_range - 1)]
                loss = -(log_probs * first_order_weighted_advantages + sum(second_order_terms)).mean().to(device)
                for i in range(self.reward_size):
                    self.recent_losses[i].append((log_probs * advantage[:,i]).mean())
                    if i != self.reward_size - 1:
                        self.recent_grads[i].append((advantage[:,i] * torch.transpose(scores,0,1)).mean(dim=1))
            else:
                loss = -(log_probs * first_order_weighted_advantages).mean().to(device)
                for i in range(self.reward_size):
                    self.recent_losses[i].append((log_probs * advantage[:,i]).mean())
        
        # Computer PPO loss
        if self.mode == 'ppo':
            with torch.no_grad():
                baseline = self.critic(states.to(device))
                outcome  = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
                advantage = (outcome - baseline).detach()
                old_log_probs = self.get_log_probs(states, actions).to(device)
                old_probs = torch.exp(old_log_probs).to(device)
            new_log_probs = self.get_log_probs(states, actions).to(device)
            new_probs = torch.exp(new_log_probs).to(device)
            ratios = (new_probs / old_probs).to(device)
            first_order_weighted_advantages = torch.sum(first_order_weights * advantage[:,0:reward_range], dim=1).to(device)
            kl_penalty = (new_log_probs - old_log_probs).to(device)
            relative_kl_weights = [self.kl_weight * first_order_weights[i] / sum(first_order_weights) for i in range(reward_range)]
            relative_kl_weights += [0.0 for _ in range(reward_range, self.reward_size)]
            # This will be slow in Pytorch because of the lack of forward-mode differentiation
            if self.second_order:
                ratio_grads = torch.stack(tuple([torch.cat([torch.flatten(g) for g in torch.autograd.grad(r.to(device), self.actor.parameters(), retain_graph=True)]) for r in ratios]), dim=0).to(device)
                kl_grads = torch.stack(tuple([torch.cat([torch.flatten(g) for g in torch.autograd.grad(kl.to(device), self.actor.parameters(), retain_graph=True)]) for kl in kl_penalty]), dim=0).to(device)
                second_order_weighted_advantages = torch.sum(second_order_weights * advantage[:,0:reward_range], dim=1).to(device)
                second_order_terms = [self.lamb[i]((ratio_grads * torch.unsqueeze(second_order_weighted_advantages, dim=1) - relative_kl_weights[i] * kl_grads).to(device)) for i in range(reward_range - 1)]
                loss =  -(ratios * first_order_weighted_advantages - self.kl_weight * kl_penalty + sum(second_order_terms)).mean().to(device)
                for i in range(self.reward_size):
                    self.recent_losses[i].append((ratios * advantage[:,i] - relative_kl_weights[i] * kl_penalty).mean())
                    if i != self.reward_size - 1:
                        self.recent_grads[i].append((ratio_grads * torch.unsqueeze(advantage[:,i], dim=1) - relative_kl_weights[i] * kl_grads).mean(dim=0))
            else:
                loss = -(ratios * first_order_weighted_advantages - self.kl_weight * kl_penalty).mean().to(device)
                for i in range(self.reward_size):
                    self.recent_losses[i].append((ratios * advantage[:,i] - relative_kl_weights[i] * kl_penalty).mean())
            # Update KL weight term as in the original PPO paper
            if kl_penalty.mean() < self.kl_target / 1.5:
                self.kl_weight *= 0.5
            else:
                self.kl_weight *= 2
        
        return loss
