# Other learning algorithms that we evaluate against
from src.TrainingParameters import TrainingParameters
from src.agents.networks import *

import math, random, pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic:

    # vanilla actor-critic

    def __init__(self, train_params: TrainingParameters, in_size, action_size, is_action_cont, hidden, extra_input=False):

        self.t = 0  # total number of frames observed
        self.discount = 0.99  # discount
        self.action_size: int = action_size
        self.is_action_cont: bool = is_action_cont

        self.batch_size: int = train_params.batch_size
        self.buffer_size: int = train_params.buffer_size
        self.no_cuda: bool = train_params.no_cuda
        self.network: str = train_params.network

        self.actor = make_network('policy', self.network, in_size, hidden, action_size, self.is_action_cont, extra_input)
        self.critic = make_network('prediction', self.network, in_size, hidden, 1, self.is_action_cont, extra_input)

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01*train_params.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=train_params.learning_rate)

        if torch.cuda.is_available() and not self.no_cuda:
            self.actor.cuda()
            self.critic.cuda()

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

        self.t += 1
        self.memory.add(state, action, reward, next_state, done)

        if self.t % self.batch_size == 0:
            self.update_critic(self.memory.sample(sample_all=True))
            self.update_actor(self.memory.sample(sample_all=True))
            self.memory.memory.clear()

    def update_actor(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        self.actor.train()

        with torch.no_grad():
            baseline = self.critic(states.to(device))
            outcome = rewards + (self.discount * self.critic(next_states.to(device)) * (1 - dones))
            advantage = (outcome - baseline).detach()

        if self.is_action_cont:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-5))
            p2 = - torch.log(torch.sqrt(2 * math.pi * variances))
            log_probs = p1 + p2
        else:
            dists = self.actor(states.to(device))
            log_probs = torch.log(torch.gather(dists, 1, actions))

        loss = -(log_probs * advantage).mean()

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

    def update(self, experiences):
        self.update_actor(experiences)
        self.update_critic(experiences)

    def save_model(self, root):
        torch.save(self.actor.state_dict(), '{}-actor.pt'.format(root))
        torch.save(self.critic.state_dict(), '{}-critic.pt'.format(root))

    def load_model(self, root):
        self.actor.load_state_dict(torch.load('{}-actor.pt'.format(root)))
        self.critic.load_state_dict(torch.load('{}-critic.pt'.format(root)))


##################################################
class DQN:

    # vanilla DQN
    # TODO - check why is_action_cont isnt used here

    def __init__(self, train_params, in_size, action_size, hidden, is_action_cont, extra_input=False):
        assert(not is_action_cont)
        self.t = 0  # total number of frames observed
        self.discount = 0.99  # discount

        self.epsilon: float = train_params.epsilon
        self.update_every: int = train_params.update_every
        self.batch_size: int = train_params.batch_size
        self.buffer_size: int = train_params.buffer_size

        self.action_size: int = action_size
        self.model = make_network('prediction', train_params.network, in_size, hidden, action_size, extra_input=extra_input)

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_params.learning_rate)

        if torch.cuda.is_available() and not train_params.no_cuda:
            self.model.cuda()

    def act(self, state):

        if np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            return random.choice(range(self.action_size))

        Q_vals = self.model(state)[0]
        return Q_vals.argmax()

    def step(self, state, action, reward, next_state, done):

        self.t += 1
        self.memory.add(state, action, reward, next_state, done)

        if self.t % self.update_every == 0 and len(self.memory) > self.batch_size:
            experience = self.memory.sample()
            self.update(experience)

    def update(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        criterion = torch.nn.MSELoss()
        self.model.train()
        predictions = self.model(states.to(device)).gather(1, actions)

        with torch.no_grad():
            predictions_next = self.model(next_states).detach()
            predictions_next = torch.stack([max(Q) for Q in torch.unbind(predictions_next, dim=0)], dim=0)
            predictions_next = predictions_next.reshape(-1, 1)
            # predictions_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)

        targets = rewards + (self.discount * predictions_next * (1 - dones))

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

class Tabular:

    # tabular Q-learning

    def __init__(self, train_params, action_size, initialisation=1):

        self.actions = list(range(action_size))
        self.Q = {}

        self.epsilon: float = train_params.epsilon
        self.alpha: float = train_params.learning_rate  # Legit

        self.discount = 0.99
        self.initialisation = initialisation

        self.t = 0

    def act(self, state):

        self.t += 1

        state = str(state)
        self.init_state(state)

        if np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            permissible_actions = self.actions
        else:
            m = max([self.Q[state][a] for a in self.actions])
            permissible_actions = [a for a in self.actions if self.Q[state][a] == m]

        action = random.choice(permissible_actions)
        return action

    def step(self, state, action, reward, next_state, done):

        state = str(state)
        next_state = str(next_state)
        self.init_state(state)
        self.init_state(next_state)

        target = reward + self.discount * max([self.Q[next_state][a] for a in self.actions]) * (1 - done)

        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * target

    def init_state(self, state):
        if state not in self.Q.keys():
            self.Q[state] = {a: self.initialisation for a in self.actions}

    def save_model(self, root):
        with open('{}-model.pt'.format(root), 'wb') as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, root):
        with open('{}-model.pt'.format(root), 'rb') as f:
            self.Q = pickle.load(f)


##################################################

class RandomAgent:

    # this agent selects actions uniformly at random

    def __init__(self, train_params, action_size, is_action_cont):
        self.is_action_cont: bool = is_action_cont
        if self.is_action_cont:
            self.number_of_actions = action_size
        else:
            self.actions = list(range(action_size))

    def act(self, state):
        if self.is_action_cont:
            return torch.tensor([random.uniform(-1, 1) for _ in range(self.number_of_actions)])
        else:
            return torch.tensor(random.choice(self.actions))

    def step(self, state, action, reward, next_state, done):
        pass

    def save_model(self, root):
        pass

    def load_model(self, root):
        pass


##################################################

class AproPO:

    # Approachability-Based Policy Optimization (for compact & convex constraints)
    # https://arxiv.org/pdf/1906.09323.pdf

    def __init__(self, train_params, in_size, action_size, hidden, is_action_cont):

        self.t = 0  # total number of frames observed
        self.eps = 0  # total number of episodes completed
        self.discount = 0.99  # discount

        self.actions = list(range(action_size))
        self.reward_size: int = train_params.reward_size
        self.action_size: int = action_size

        self.is_action_cont: bool = is_action_cont
        self.batch_size: int = train_params.batch_size
        self.buffer_size: int = train_params.buffer_size
        self.update_every_eps = train_params.update_every_eps
        self.lambda_lr_2 = train_params.lambda_lr_2
        self.no_cuda: bool = train_params.no_cuda

        self.constraints = train_params.constraints  # [(lower, upper),(lower, upper)...]

        # can be generalised to arbitrary compact & convex constraints by amending max_dist below
        # and also the projection function
        max_dist = sum([(u - l) ** self.reward_size for (l, u) in self.constraints]) ** (1.0 / self.reward_size)
        tolerance = 0.01
        self.kappa = max_dist / (2 * tolerance) ** 0.5
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.best_response_oracle = ActorCritic(train_params,
                                                in_size=in_size,
                                                action_size=action_size,
                                                hidden=hidden,
                                                is_action_cont=is_action_cont,
                                                extra_input=True)

        # self.best_response_oracle = DQN(in_size, action_size, network, hidden, extra_input=True)

        self.lamb = np.asarray([random.random() for _ in range(self.reward_size)])
        self.cumulative = self.kappa

    def act(self, state):

        state = self.augment(state, self.cumulative)

        return self.best_response_oracle.act(state)

    def step(self, state, action, reward, next_state, done):

        self.t += 1

        state = self.augment(state, self.cumulative)
        next_state = self.augment(next_state, self.discount * self.cumulative)

        self.cumulative = self.discount * self.cumulative

        self.memory.add(state, action, reward, next_state, done)

        if done:

            self.eps += 1
            self.cumulative = self.kappa

            if self.eps % self.update_every_eps == 0 and len(self.memory) > self.batch_size:
                experience = self.memory.sample(sample_all=True)
                self.update(experience)

    def augment(self, state, kappa):

        dtype = torch.cuda.FloatTensor if (torch.cuda.is_available() and not self.no_cuda) else torch.FloatTensor
        t = torch.cat((state.flatten(), torch.tensor([kappa]).type(dtype)), 0).unsqueeze(0)

        return t

    def project(self, x):

        # project x onto constraint cube
        p = np.zeros(self.reward_size)
        for i in range(self.reward_size):
            l, u = self.constraints[i]
            if x[i] < l:
                p[i] = l
            elif x[i] > u:
                p[i] = u
            else:
                p[i] = x[i]

        # project this onto target set
        projection = (x - p) / max(1, np.linalg.norm(x - p, 2))

        return projection

    def update(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        # update best_response_oracle with normal learning

        rewards2 = torch.tensor([-np.dot(self.lamb, r.cpu()) for r in rewards]).reshape(-1, 1).float().to(device)
        self.best_response_oracle.update((states, actions, rewards2, next_states, dones))

        # get gradient for lambda and update with OGD

        lamb_grad = np.zeros(self.reward_size)
        ep_return = np.zeros(self.reward_size)
        eps = 0

        for reward, done in list(zip(rewards, dones))[::-1]:

            if done:
                eps += 1
                lamb_grad += ep_return
                ep_return = np.zeros(self.reward_size)

            ep_return = np.asarray(reward.cpu()) + self.discount * ep_return

        lamb_grad += ep_return
        lamb_grad /= eps  # dones.sum() # counts the episodes

        self.lamb = self.project(self.lamb + self.lambda_lr_2 * lamb_grad)

    def save_model(self, root):
        self.best_response_oracle.save_model(root)

    def load_model(self, root):
        self.best_response_oracle.load_model(root)


##################################################

class RCPO:

    # Reward-Constrained Policy Optimisation
    # https://arxiv.org/pdf/1805.11074.pdf

    def __init__(self, train_params, action_size, in_size, hidden, is_action_cont):

        self.t = 0  # total number of frames observed
        self.discount = 0.99  # discount
        self.constraint = train_params.constraint

        self.action_size: int = action_size
        self.is_action_cont: bool = is_action_cont

        self.actor = make_network('policy', train_params.network, in_size, hidden, action_size, self.is_action_cont)
        self.critic = make_network('prediction', train_params.network, in_size, hidden, 1, self.is_action_cont)

        self.buffer_size: int = train_params.buffer_size

        self.batch_size: int = train_params.batch_size
        self.no_cuda: bool = train_params.no_cuda

        self.lamb = random.random()
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.LAMBDA_LR = 0.1*train_params.learning_rate

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01*train_params.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=train_params.learning_rate)

        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        if torch.cuda.is_available() and not self.no_cuda:
            self.actor.cuda()
            self.critic.cuda()

    def act(self, state):

        if self.is_action_cont:
            mu, var = self.actor(state)
            mu = mu.data.cpu().numpy()
            sigma = torch.sqrt(var).data.cpu().numpy()
            action = np.random.normal(mu, sigma)
            return torch.tensor(np.clip(action, -1, 1))
        else:
            return Categorical(self.actor(state)).sample()

    def step(self, state, action, rewards, next_state, done):

        self.t += 1
        self.memory.add(state, action, rewards, next_state, done)

        # if done:
        if self.t % self.batch_size == 0:
            experiences = self.memory.sample(sample_all=True)

            self.update_critic(experiences)
            self.update_actor(experiences)
            self.update_lagrange(experiences)

            self.memory.memory.clear()

    def update_actor(self, experiences):

        states, actions, rewards, next_states, dones = experiences
        rewards = (rewards[:, 0] - self.lamb * rewards[:, 1]).unsqueeze(-1)

        self.actor.train()

        with torch.no_grad():
            baseline = self.critic(states.to(device))
            outcome = rewards + (self.discount * self.critic(next_states.to(device)) * (1 - dones))
            advantage = (outcome - baseline).detach()

        if self.is_action_cont:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-5))
            p2 = - torch.log(torch.sqrt(2 * math.pi * variances))
            log_probs = p1 + p2
        else:
            dists = self.actor(states.to(device))
            log_probs = torch.log(torch.gather(dists, 1, actions))

        loss = -(log_probs * advantage).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        self.actor.eval()

    def update_critic(self, experiences):

        states, _, rewards, next_states, dones = experiences
        rewards = (rewards[:, 0] - self.lamb * rewards[:, 1]).reshape(-1, 1)

        self.critic.train()

        prediction = self.critic(states.to(device))
        with torch.no_grad():
            target = rewards + (self.discount * self.critic(next_states.to(device)) * (1 - dones))

        loss = nn.MSELoss()(prediction, target).to(device)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        self.critic.eval()

    def update_lagrange(self, experiences):

        states, _, rewards, next_states, dones = experiences
        cost = sum(rewards[:, 1])
        self.lamb += self.LAMBDA_LR * (cost - self.constraint)
        self.lamb = max(self.lamb, 0)

    def save_model(self, root):
        torch.save(self.actor.state_dict(), '{}-actor.pt'.format(root))
        torch.save(self.critic.state_dict(), '{}-critic.pt'.format(root))

    def load_model(self, root):
        self.actor.load_state_dict(torch.load('{}-actor.pt'.format(root)))
        self.critic.load_state_dict(torch.load('{}-critic.pt'.format(root)))


##################################################

class VaR_AC:

    # Policy Gradient for VaR-constrained problems
    # https://stanfordasl.github.io/wp-content/papercite-data/pdf/Chow.Ghavamzadeh.Janson.Pavone.JMLR18.pdf

    def __init__(self, train_params, action_size, in_size, hidden, is_action_cont):

        self.batch_size: int = train_params.batch_size
        self.buffer_size: int = train_params.buffer_size
        self.no_cuda: bool = train_params.no_cuda

        self.t = 0  # total number of frames observed
        self.cumulative_cost = 0

        # TODO - consider moving these to train_params
        self.discount = 0.99  # discount
        self.LAMBDA_LR = train_params.learning_rate

        self.alpha: float = train_params.alpha  # constraint
        self.beta: float = train_params.beta  # tolerance


        self.action_size: int = action_size
        self.is_action_cont: bool = is_action_cont

        self.actor = make_network('policy', train_params.network, in_size, hidden, action_size, is_action_cont, extra_input=True)
        self.critic = make_network('prediction', train_params.network, in_size, hidden, 1, is_action_cont, extra_input=True)

        self.lamb = random.random()

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01*train_params.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=train_params.learning_rate)

        if torch.cuda.is_available() and not self.no_cuda:
            self.actor.cuda()
            self.critic.cuda()

    def act(self, state):

        state = self.augment(state, self.cumulative_cost)

        if self.is_action_cont:
            mu, var = self.actor(state)
            mu = mu.data.cpu().numpy()
            sigma = torch.sqrt(var).data.cpu().numpy()
            action = np.random.normal(mu, sigma)
            return torch.tensor(np.clip(action, -1, 1))
        else:
            return Categorical(self.actor(state)).sample()

    def augment(self, state, cost):

        dtype = torch.cuda.FloatTensor if (torch.cuda.is_available() and not self.no_cuda) else torch.FloatTensor
        t = torch.cat((state.flatten(), torch.tensor([cost]).type(dtype)), 0).unsqueeze(0)

        return t

    def step(self, state, action, rewards, next_state, done):

        self.t += 1

        state = self.augment(state, self.cumulative_cost)
        next_state = self.augment(next_state, self.cumulative_cost + rewards[1])

        self.cumulative_cost += rewards[1]

        if not done:
            r = rewards[0]
        else:
            r = self.lamb if self.cumulative_cost < 0 else 0
            self.cumulative_cost = 0

        self.memory.add(state, action, r, next_state, done)

        if done:
            experiences = self.memory.sample(sample_all=True)

            self.update_critic(experiences)
            self.update_actor(experiences)
            self.update_lagrange(experiences)

            self.memory.memory.clear()

    def update_actor(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        self.actor.train()

        with torch.no_grad():
            baseline = self.critic(states.to(device))
            outcome = rewards + (self.discount * self.critic(next_states.to(device)) * (1 - dones))
            advantage = (outcome - baseline).detach()

        if self.is_action_cont:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-5))
            p2 = - torch.log(torch.sqrt(2 * math.pi * variances))
            log_probs = p1 + p2
        else:
            dists = self.actor(states.to(device))
            log_probs = torch.log(torch.gather(dists, 1, actions))

        loss = -(log_probs * advantage).mean()

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

    def update_lagrange(self, experiences):

        states, _, rewards, next_states, dones = experiences
        # cost = 1 if sum(rewards[:,1]) > self.alpha else 07
        cost = 1 if states[-1][-1] < 0 else 0
        self.lamb += self.LAMBDA_LR * (cost - self.beta)
        self.lamb = max(self.lamb, 0)

    def save_model(self, root):
        torch.save(self.actor.state_dict(), '{}-actor.pt'.format(root))
        torch.save(self.critic.state_dict(), '{}-critic.pt'.format(root))

    def load_model(self, root):
        self.actor.load_state_dict(torch.load('{}-actor.pt'.format(root)))
        self.critic.load_state_dict(torch.load('{}-critic.pt'.format(root)))


##################################################

# Currently untested

# class VaR_PG:
#
#     # Policy Gradient for VaR-constrained problems
#     # https://stanfordasl.github.io/wp-content/papercite-data/pdf/Chow.Ghavamzadeh.Janson.Pavone.JMLR18.pdf
#
#     def __init__(self, action_size, alpha, beta, in_size=4, network='DNN', hidden, is_action_cont=False):
#
#         self.t = 0  # total number of frames observed
#         self.discount = 0.99  # discount
#
#         self.alpha: float = train_params.alpha  # constraint
#         self.beta: float = train_params.beta  # tolerance
#         self.lamb = random.random()
#
#         self.LAMBDA_LR = 0.01
#
#         self.action_size: int = action_size
#         self.is_action_cont: bool = is_action_cont
#
#         self.actor = make_network('policy', network, in_size, hidden, action_size, is_action_cont)
#
#         self.actor_optimizer = optim.Adam(self.actor.parameters())
#
#         self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
#
#         if torch.cuda.is_available() and not self.no_cuda:
#             self.actor.cuda()
#
#     def act(self, state):
#
#         if self.is_action_cont:
#             mu, var = self.actor(state)
#             mu = mu.data.cpu().numpy()
#             sigma = torch.sqrt(var).data.cpu().numpy()
#             action = np.random.normal(mu, sigma)
#             return torch.tensor(np.clip(action, -1, 1))
#         else:
#             return Categorical(self.actor(state)).sample()
#
#     def step(self, state, action, rewards, next_state, done):
#
#         self.t += 1
#         self.memory.add(state, action, rewards, next_state, done)
#
#         if self.t % self.buffer_size == 0:
#             experiences = self.memory.sample(sample_all=True)
#
#             self.update_actor(experiences)
#             self.update_lagrange(experiences)
#
#             self.memory.memory.clear()
#
#     def update_actor(self, experiences):
#
#         states, actions, rewards, next_states, dones = experiences
#
#         reward = sum(rewards[:, 0])
#         cost = 1 if sum(rewards[:, 1]) > self.alpha else 0
#
#         self.actor.train()
#
#         if self.is_action_cont:
#             means, variances = self.actor(states.to(device))
#             p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-5))
#             p2 = - torch.log(torch.sqrt(2 * math.pi * variances))
#             log_probs = p1 + p2
#         else:
#             dists = self.actor(states.to(device))
#             log_probs = torch.log(torch.gather(dists, 1, actions))
#
#         loss = -(log_probs * (reward + self.lamb * cost)).mean()
#
#         self.actor_optimizer.zero_grad()
#         loss.backward()
#         self.actor_optimizer.step()
#
#         self.actor.eval()
#
#     def update_lagrange(self, experiences):
#
#         states, _, rewards, next_states, dones = experiences
#         cost = 1 if sum(rewards[:, 1]) > self.alpha else 0
#         self.lamb += self.LAMBDA_LR * (cost - self.beta)
#         self.lamb = max(self.lamb, 0)
#
#     def save_model(self, root):
#         torch.save(self.actor.state_dict(), '{}-actor.pt'.format(root))
#
#     def load_model(self, root):
#         self.actor.load_state_dict(torch.load('{}-actor.pt'.format(root)))