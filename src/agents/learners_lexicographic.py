# Our (lexicographic) learning algorithms

from src.agents.networks import *

import math
import random
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical


class LexDQN:

    # lexicographic DQN

    def __init__(self, train_params, in_size, action_size, hidden):
        self.update_steps = train_params.update_steps
        self.epsilon = train_params.epsilon
        self.buffer_size = train_params.buffer_size
        self.batch_size = train_params.batch_size
        self.no_cuda = train_params.no_cuda
        self.update_every = train_params.update_every
        self.slack = train_params.slack
        self.reward_size = train_params.reward_size
        self.network = train_params.network

        self.t = 0  # total number of frames observed
        self.discount = 0.99  # discount

        self.actions = list(range(action_size))
        self.action_size = action_size

        if self.network == 'DNN':
            self.model = DNN(in_size, (self.reward_size, action_size), hidden)
        elif self.network == 'CNN':
            self.model = CNN(int((in_size / 3) ** 0.5), channels=3,
                             out_size=(self.reward_size, action_size),
                             convs=hidden, hidden=hidden)
        else:
            print('invalid network specification')
            assert False

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_params.learning_rate)

        if torch.cuda.is_available() and not self.no_cuda:
            self.model.cuda()

    def act(self, state):

        if np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            return random.choice(self.actions)

        Q = self.model(state)[0]
        permissible_actions = self.permissible_actions(Q)
        return random.choice(permissible_actions)

    def permissible_actions(self, Q):
        permissible_actions = self.actions

        for i in range(self.reward_size):
            Qi = Q[i, :]
            m = max([Qi[a] for a in permissible_actions])
            r = self.slack
            permissible_actions = [a for a in permissible_actions if Qi[a] >= m - r * abs(m)]

        return permissible_actions

    def lexmax(self, Q):

        a = self.permissible_actions(Q)[0]
        return Q[:, a]

    def step(self, state, action, reward, next_state, done):

        self.t += 1
        self.memory.add(state, action, reward, next_state, done)

        if self.t % self.update_every == 0 and len(self.memory) > self.batch_size:
            experience = self.memory.sample()
            for _ in range(self.update_steps):
                self.update(experience)

    def update(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        criterion = torch.nn.MSELoss()
        self.model.train()

        # probably slightly suboptimal
        idx = torch.cat((actions, actions), 1).reshape(-1, self.reward_size, 1)
        predictions = self.model(states.to(device)).gather(2, idx).squeeze()

        with torch.no_grad():
            predictions_next = self.model(next_states).detach()
            next_values = torch.stack([self.lexmax(Q) for Q in torch.unbind(predictions_next, dim=0)], dim=0)

        targets = rewards + (self.discount * next_values * (1 - dones))

        loss = criterion(predictions, targets).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.model.eval()

    def save_model(self, root):
        torch.save(self.model.state_dict(), '{}-model.pt'.format(root))

    def load_model(self, root):
        self.model.load_state_dict(torch.load('{}-model.pt'.format(root)))


##################################################

class LexActorCritic:

    def __init__(self, train_params, in_size, action_size, mode, is_action_cont,
                 second_order, sequential, extra_input, hidden):

        self.no_cuda = train_params.no_cuda
        self.batch_size = train_params.batch_size
        self.buffer_size = train_params.buffer_size
        self.network = train_params.network
        self.reward_size = train_params.reward_size

        self.action_size = action_size
        self.mode = mode
        self.is_action_cont = is_action_cont

        if mode != 'a2c' and mode != 'ppo':
            print("Error: mode must be \'a2c\' or \'ppo\'")
            return

        self.t = 0  # total number of frames observed
        self.discount = 0.99  # discount

        self.actor = make_network('policy', self.network, in_size, hidden, self.action_size, is_action_cont,
                                  extra_input)
        self.critic = make_network('prediction', self.network, in_size, hidden, self.reward_size, is_action_cont,
                                   extra_input)
        self.mu = [0.0 for _ in range(self.reward_size - 1)]
        self.j = [0.0 for _ in range(self.reward_size - 1)]
        self.recent_losses = [collections.deque(maxlen=50) for i in range(self.reward_size)]

        # If updating the actor sequentially (one objective at a time) we need to keep track of which objective we're using
        self.i = 0 if sequential else None

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # self.actor_optimizer = optim.Adam(self.actor.parameters())
        # self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=train_params.learning_rate * 0.01)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=train_params.learning_rate)

        # If using Adam then only the eta LR and the (relative) beta LRs matter
        self.beta = list(reversed(range(1, self.reward_size + 1)))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, self.reward_size + 1)))]

        # If using second order terms
        self.second_order = second_order
        if self.second_order:
            actor_params = filter(lambda p: p.requires_grad, self.actor.parameters())
            self.lamb = [LagrangeLambda(tuple([p.shape for p in actor_params])) for _ in range(self.reward_size - 1)]
            self.lagrange_optimizer = [optim.Adam(ll.parameters()) for ll in self.lamb]
            self.grad_j = [0.0 for _ in range(self.reward_size - 1)]
            self.recent_grads = [collections.deque(maxlen=25) for i in range(self.reward_size - 1)]

        if (torch.cuda.is_available() and not self.no_cuda):
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
        if self.is_action_cont:
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

        if self.t % self.batch_size == 0:
            self.update(self.memory.sample(sample_all=True))
            self.memory.memory.clear()

    def update_actor(self, experiences):

        self.actor.train()

        loss = self.compute_loss(experiences, self.i + 1) if self.i != None else self.compute_loss(experiences,
                                                                                                   self.reward_size)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.actor.eval()

    def update_critic(self, experiences):

        states, _, rewards, next_states, dones = experiences

        self.critic.train()

        prediction = self.critic(states.to(device))
        with torch.no_grad():
            target = rewards + (self.discount * self.critic(next_states.to(device)) * (1 - dones))

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
                    self.j[self.i] = -torch.tensor(self.recent_losses[self.i][25:]).mean()
                    if self.second_order:
                        self.grad_j[self.i] = -torch.stack(tuple(self.recent_grads[self.i]), dim=0).mean(dim=0)
            else:
                # print("Converged for reward function {}!".format(self.i))
                self.i = 0 if self.i == self.reward_size - 1 else self.i + 1

        else:
            for i in range(self.reward_size - 1):
                self.j[i] = -torch.tensor(self.recent_losses[i][25:]).mean()
                if self.second_order:
                    self.grad_j[i] = -torch.stack(tuple(self.recent_grads[i]), dim=0).mean(dim=0)

        # Update Lagrange parameters
        r = self.i if self.i != None else self.reward_size - 1
        for i in range(r):
            self.mu[i] += self.eta[i] * (self.j[i] - (-self.recent_losses[i][-1]))
            self.mu[i] = max(self.mu[i], 0.0)
            if self.second_order:
                self.lamb[i].train()
                loss = self.lamb[i](torch.unsqueeze(self.grad_j[i] - (-self.recent_grads[i][-1], 0)).to(device)).to(device)
                self.lagrange_optimizer[i].zero_grad()
                loss.backward()
                self.lagrange_optimizer[i].step()
                self.lamb[i].eval()

    def update(self, experiences):
        self.update_actor(experiences)
        self.update_critic(experiences)
        self.update_lagrange()

    # If updating sequentially, check whether previous loss has converged
    def converged(self, tolerance=0.1, minimum_updates=50):

        # If not enough updates have been performed, assume not converged
        if len(self.recent_losses[self.i]) < minimum_updates:
            return False
        # Else check if the loss has stopped improving, within some tolerance
        else:
            l_old_mean = torch.tensor(self.recent_losses[self.i][:24]).mean().float()
            l_new_mean = torch.tensor(self.recent_losses[self.i][25:]).mean().float()
            if abs(l_old_mean - l_new_mean)/abs(l_new_mean) > tolerance:
                return False
        
        return True

        return True

    def get_log_probs(self, states, actions):

        if self.is_action_cont:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-3))
            p2 = - torch.log(torch.sqrt(2 * math.pi * variances))
            log_probs = p1 + p2
        else:
            dists = self.actor(states.to(device))
            log_probs = torch.log(torch.gather(dists, 1, actions))

        return torch.sum(log_probs, dim=1)

    def compute_loss(self, experiences, reward_range):

        states, actions, rewards, next_states, dones = experiences
        loss = "Undefined"
        second_order_weights = "Undefined"
        # Compute weights for different components of the first order objective terms
        first_order = []
        for i in range(reward_range - 1):
            if self.i != None:
                w = self.beta[reward_range - 1] * self.mu[i]
            else:
                w = self.beta[i] + self.mu[i] * sum([self.beta[j] for j in range(i + 1, reward_range)])
            first_order.append(w)
        first_order.append(self.beta[reward_range - 1])
        first_order_weights = torch.tensor(first_order)

        # If needed, compute the weights for the second order objective terms as well
        if self.second_order:
            if self.i != None:
                second_order = [self.beta[self.i] for _ in range(reward_range - 1)]
            else:
                second_order = [sum([self.beta[j] for j in range(i + 1, reward_range)]) for i in
                                range(reward_range - 1)]
                second_order.append(self.beta[reward_range - 1])
            second_order_weights = torch.tensor(second_order)

        # Computer A2C loss
        if self.mode == 'a2c':
            with torch.no_grad():
                baseline = self.critic(states.to(device))
                outcome = rewards + (
                        self.discount * self.critic(next_states.to(device))[:, 0:reward_range] * (1 - dones))
                advantage = (outcome - baseline).detach()
            log_probs = self.get_log_probs(states, actions)
            first_order_weighted_advantages = torch.sum(first_order_weights * advantage[:, 0:reward_range], dim=1).to(
                device)
            # This will be slow in Pytorch because of the lack of forward-mode differentiation
            if self.second_order:

                if second_order_weights == "undefined":
                    raise Exception("second_order_weights is never defined")

                scores = torch.stack(tuple([torch.cat([torch.flatten(g) for g in
                                                       torch.autograd.grad(l_p.to(device), self.actor.parameters(),
                                                                           retain_graph=True)]) for l_p in log_probs]),
                                     dim=0).to(device)
                second_order_weighted_advantages = torch.sum(second_order_weights * advantage[:, 0:reward_range],
                                                             dim=1).to(device)
                second_order_terms = [self.lamb[i](scores.to(device)) * second_order_weighted_advantages for i in
                                      range(reward_range - 1)]
                loss = -(log_probs * first_order_weighted_advantages + sum(second_order_terms)).mean().to(device)
                for i in range(self.reward_size):
                    self.recent_losses[i].append(-(log_probs * advantage[:,i]).mean())
                    if i != self.reward_size - 1:
                        self.recent_grads[i].append(-(advantage[:,i] * torch.transpose(scores,0,1)).mean(dim=1))
            else:
                loss = -(log_probs * first_order_weighted_advantages).mean().to(device)
                for i in range(self.reward_size):
                    self.recent_losses[i].append(-(log_probs * advantage[:,i]).mean())
        
        # Computer PPO loss
        if self.mode == 'ppo':
            with torch.no_grad():
                baseline = self.critic(states.to(device))
                outcome = rewards + (self.discount * self.critic(next_states.to(device)) * (1 - dones))
                advantage = (outcome - baseline).detach()
                old_log_probs = self.get_log_probs(states, actions).to(device)
            new_log_probs = self.get_log_probs(states, actions).to(device)
            ratios = torch.exp(new_log_probs - old_log_probs).to(device)
            first_order_weighted_advantages = torch.sum(first_order_weights * advantage[:, 0:reward_range], dim=1).to(
                device)
            kl_penalty = (new_log_probs - old_log_probs).to(device)
            relative_kl_weights = [self.kl_weight * first_order_weights[i] / sum(first_order_weights) for i in
                                   range(reward_range)]
            relative_kl_weights += [0.0 for _ in range(reward_range, self.reward_size)]
            # This will be slow in Pytorch because of the lack of forward-mode differentiation
            if self.second_order:
                ratio_grads = torch.stack(tuple([torch.cat([torch.flatten(g) for g in
                                                            torch.autograd.grad(r.to(device), self.actor.parameters(),
                                                                                retain_graph=True)]) for r in ratios]),
                                          dim=0).to(device)
                kl_grads = torch.stack(tuple([torch.cat([torch.flatten(g) for g in
                                                         torch.autograd.grad(kl.to(device), self.actor.parameters(),
                                                                             retain_graph=True)]) for kl in
                                              kl_penalty]), dim=0).to(device)
                second_order_weighted_advantages = torch.sum(second_order_weights * advantage[:, 0:reward_range],
                                                             dim=1).to(device)
                second_order_terms = [self.lamb[i]((ratio_grads * torch.unsqueeze(second_order_weighted_advantages,
                                                                                  dim=1) - relative_kl_weights[
                                                        i] * kl_grads).to(device)) for i in range(reward_range - 1)]
                loss = -(ratios * first_order_weighted_advantages - self.kl_weight * kl_penalty + sum(
                    second_order_terms)).mean().to(device)
                for i in range(self.reward_size):
                    self.recent_losses[i].append(-(ratios * advantage[:,i] - relative_kl_weights[i] * kl_penalty).mean())
                    if i != self.reward_size - 1:
                        self.recent_grads[i].append(-(ratio_grads * torch.unsqueeze(advantage[:,i], dim=1) - relative_kl_weights[i] * kl_grads).mean(dim=0))
            else:
                loss = -(ratios * first_order_weighted_advantages - self.kl_weight * kl_penalty).mean().to(device)
                for i in range(self.reward_size):
                    self.recent_losses[i].append(-(ratios * advantage[:,i] - relative_kl_weights[i] * kl_penalty).mean())
                
                # n = torch.tensor(float('nan'))
                # m = torch.tensor(float('inf'))
                # o = torch.tensor(float('-inf'))

                # Check for nans and infs
                if torch.isnan(loss) or torch.isinf(loss):
                    current_time = str(time.time())
                    torch.save(self.actor.state_dict(), 'actor_nan_error_{}.pt'.format(current_time))
                    torch.save(self.critic.state_dict(), 'critic_nan_error_{}.pt'.format(current_time))
                    current_data = {'loss': loss, 'experiences': experiences, 'ratios': ratios,
                                    'first_order_weights': first_order_weights, 'advantage': advantage,
                                    'kl_weight': self.kl_weight, 'kl_penalty': kl_penalty}
                    with open('nan_error_data_{}.pickle'.format(current_time), 'wb') as handle:
                        pickle.dump(current_data, handle)
                    return "oops"

            # Update KL weight term as in the original PPO paper
            if kl_penalty.mean() < self.kl_target / 1.5:
                self.kl_weight *= 0.5
            elif kl_penalty.mean() > self.kl_target * 1.5:
                self.kl_weight *= 2
        if loss == "undefined":
            raise Exception("Loss is never defined")
        return loss

    def save_model(self, root):
        torch.save(self.actor.state_dict(), '{}-actor.pt'.format(root))
        torch.save(self.critic.state_dict(), '{}-critic.pt'.format(root))

    def load_model(self, root):
        self.actor.load_state_dict(torch.load('{}-actor.pt'.format(root)))
        self.critic.load_state_dict(torch.load('{}-critic.pt'.format(root)))

#################################################


class LexTabular:

    def __init__(self, train_params, in_size, action_size, initialisation, double=False):

        self.slack = train_params.slack
        self.epsilon = train_params.epsilon

        self.actions = list(range(action_size))
        if not double:
            self.Q = [{} for _ in range(train_params.reward_size)]
        else:
            self.Qa = [{} for _ in range(train_params.reward_size)]
            self.Qb = [{} for _ in range(train_params.reward_size)]

        self.discount = 0.99

        if isinstance(initialisation, float) or isinstance(initialisation, int):
            self.initialisation = [initialisation for _ in range(train_params.reward_size)]
        else:
            self.initialisation = initialisation

        self.double = double
        if double:
            self.step = self.double_Q_update
        elif train_params.lextab_on_policy:
            self.step = self.SARSA_update
        else:
            self.step = self.Q_update

    def act(self, state):

        state = str(state)
        self.init_state(state)
        return self.lexicographic_epsilon_greedy(state)

    def init_state(self, state):

        if not self.double:
            if state not in self.Q[0].keys():
                for i, Qi in enumerate(self.Q):
                    Qi[state] = {a: self.initialisation[i] for a in self.actions}  # initialisation
        else:
            if state not in self.Qa[0].keys():
                for i, Qai in enumerate(self.Qa):
                    Qai[state] = {a: self.initialisation[i] for a in self.actions}  # initialisation
                for i, Qbi in enumerate(self.Qb):
                    Qbi[state] = {a: self.initialisation[i] for a in self.actions}  # self.initialisation

    def lexicographic_epsilon_greedy(self, state):

        state = str(state)

        if np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            return np.random.choice(self.actions)

        permissible_actions = self.actions

        if not self.double:
            for Qi in self.Q:
                m = max([Qi[state][a] for a in permissible_actions])
                r = self.slack
                permissible_actions = [a for a in permissible_actions if Qi[state][a] >= m - r * abs(m)]
        else:
            for Qai, Qbi in zip(self.Qa, self.Qb):
                m = max([0.5 * (Qai[state][a] + Qbi[state][a]) for a in permissible_actions])
                r = self.slack
                permissible_actions = [a for a in permissible_actions if
                                       0.5 * (Qai[state][a] + Qbi[state][a] >= m - r * abs(m))]

        return np.random.choice(permissible_actions)

    def Q_update(self, state, action, reward, next_state, done):

        state = str(state)
        next_state = str(next_state)
        self.init_state(state)
        self.init_state(next_state)
        permissible_actions = self.actions

        for i, Qi in enumerate(self.Q):
            m = max([Qi[next_state][a] for a in permissible_actions])
            r = self.slack
            permissible_actions = [a for a in permissible_actions if Qi[next_state][a] >= m - r * abs(m)]

            alpha = 0.01
            Qi[state][action] = (1 - alpha) * Qi[state][action] + alpha * (reward[i] + self.discount * m)

    def SARSA_update(self, state, action, reward, next_state, done):

        state = str(state)
        next_state = str(next_state)
        permissible_actions = self.actions

        for Qi in self.Q:
            m = max([Qi[next_state][a] for a in permissible_actions])
            r = self.slack
            permissible_actions = [a for a in permissible_actions if Qi[next_state][a] >= m - r * abs(m)]

        ps = []

        for Qi in self.Q:
            for a in self.actions:
                if a in permissible_actions:
                    ps.append((1 - self.epsilon) / len(permissible_actions) + self.epsilon / len(self.actions))
                else:
                    ps.append(self.epsilon / len(self.actions))

        for i, Qi in enumerate(self.Q):
            exp = sum([p * Qi[next_state][a] for p, a in zip(ps, self.actions)])
            target = reward[i] + self.discount * exp
            alpha = 0.01
            Qi[state][action] = (1 - alpha) * Qi[state][action] + alpha * target

    def double_Q_update(self, state, action, reward, next_state, done):

        state = str(state)
        next_state = str(next_state)
        permissible_actions = self.actions
        r = self.slack

        for i, (Qai, Qbi) in enumerate(zip(self.Qa, self.Qb)):

            if np.random.choice([True, False]):

                m = max([Qbi[next_state][a] for a in permissible_actions])
                permissible_actions = [a for a in permissible_actions if Qbi[next_state][a] >= m - r * abs(m)]

                a = np.random.choice(permissible_actions)
                m = Qai[next_state][a]
                target = 0 if done else reward[i] + self.discount * m

                alpha = 0.01
                Qai[state][action] = (1 - alpha) * Qai[state][action] + alpha * target

            else:

                m = max([Qai[next_state][a] for a in permissible_actions])
                permissible_actions = [i for i in permissible_actions if Qai[next_state][i] >= m - r * abs(m)]

                a = np.random.choice(permissible_actions)
                m = Qbi[next_state][a]
                target = 0 if done else reward[i] + self.discount * m

                alpha = 0.01
                Qbi[state][action] = (1 - alpha) * Qbi[state][action] + alpha * target

    def save_model(self, root):
        if not self.double:
            with open('{}-model.pt'.format(root), 'wb') as f:
                pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open('{}-model-A.pt'.format(root), 'wb') as f:
                pickle.dump(self.Qa, f, pickle.HIGHEST_PROTOCOL)
            with open('{}-model-B.pt'.format(root), 'wb') as f:
                pickle.dump(self.Qb, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, root):
        if not self.double:
            with open('{}-model.pt'.format(root), 'rb') as f:
                self.Q = pickle.load(f)
        else:
            with open('{}-model-A.pt'.format(root), 'rb') as f:
                self.Qa = pickle.load(f)
            with open('{}-model-B.pt'.format(root), 'rb') as f:
                self.Qb = pickle.load(f)
