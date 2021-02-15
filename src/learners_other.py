# Other learning algorithms that we evaluate against

class ActorCritic:
    
    # vanilla actor-critic
    
    def __init__(self, in_size, action_size, network='DNN', hidden=16, continuous=False, extra_input=False):
            
        self.t = 0                                   # total number of frames observed
        self.discount = 0.99                         # discount
        
        self.action_size = action_size
        self.continuous = continuous
        
        self.actor = make_network('policy', network, in_size, hidden, action_size, continuous, extra_input)
        self.critic = make_network('prediction', network, in_size, hidden, 1, continuous, extra_input)
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
        
    
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
        
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)
        
        #if done:
        if self.t % BATCH_SIZE == 0:
            self.update_critic(self.memory.sample(sample_all=True))
            self.update_actor(self.memory.sample(sample_all=True))
            self.memory.memory.clear()

            
    def update_actor(self, experiences):
 
        states, actions, rewards, next_states, dones = experiences
    
        self.actor.train()
        
        with torch.no_grad():
            baseline = self.critic(states.to(device))
            outcome  = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
            advantage = (outcome - baseline).detach()
        
        if self.continuous:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-3))
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
            target = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
            
        loss = nn.MSELoss()(prediction, target).to(device)
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        self.critic.eval()
        

    def update(self, experiences):
        self.update_actor(experiences)
        self.update_critic(experiences)

##################################################

class DQN:
    
    # vanilla DQN
    
    def __init__(self, in_size, action_size, network='DNN', hidden=16):
            
        self.t = 0                                   # total number of frames observed
        self.discount = 0.99                         # discount
        
        self.action_size=action_size
        self.model = make_network('prediction', network, in_size, hidden, action_size)
            
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.optimizer = optim.Adam(self.model.parameters())
        
        if torch.cuda.is_available():
            self.model.cuda()
        
    
    def act(self, state):

        if np.random.choice([True,False], p=[EPSILON, 1-EPSILON]):
            return random.choice(range(self.action_size))
            
        Q_vals = self.model(state)[0]
        return Q_vals.argmax()
    
    
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
        predictions = self.model(states.to(device)).gather(1,actions)
    
        with torch.no_grad():
            predictions_next = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        
        targets = rewards + (self.discount * predictions_next * (1-dones))
        
        loss = criterion(predictions, targets).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.model.eval()

##################################################

class Tabular:
    
    # tabular Q-learning
    
    def __init__(self, action_size, initialisation=1):
        
        self.actions = list(range(action_size))
        self.Q = {}
        
        self.discount = 0.99
        self.initialisation = initialisation
        
        self.t = 0
    
    def act(self, state):
        
        self.t += 1
        
        state = str(state)
        self.init_state(state)
            
        if np.random.choice([True,False], p=[EPSILON, 1-EPSILON]):
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

        target = reward + self.discount * max([self.Q[next_state][a] for a in self.actions]) * (1-done)
        
        alpha = 0.05
        self.Q[state][action] = (1 - alpha) * self.Q[state][action] + alpha * target
        
    def init_state(self, state):
        if not state in self.Q.keys(): 
            self.Q[state] = {a:self.initialisation for a in self.actions}

##################################################
            
class RandomAgent:
    
    # this agent selects actions uniformly at random

    def __init__(self, action_size, continuous=False):
        self.continuous = continuous
        if continuous:
            self.number_of_actions = action_size
        else:
            self.actions = list(range(action_size))
            
    def act(self, state):
        if self.continuous:
            return torch.tensor([random.uniform(-1, 1) for _ in range(self.number_of_actions)])
        else:
            return torch.tensor(random.choice(self.actions))
    
    def step(self, state, action, reward, next_state, done):
        pass

##################################################

class AproPO:
    
    # Approachability-Based Policy Optimization (for compact & convex constraints)
    # https://arxiv.org/pdf/1906.09323.pdf
    
    def __init__(self, in_size, action_size, constraints, reward_size=2, 
                 network='DNN', hidden=16, continuous=False):
            
        self.t = 0                                   # total number of frames observed
        self.eps = 0                                 # total number of episodes completed
        self.discount = 0.99                         # discount
        
        self.actions = list(range(action_size))
        self.reward_size = reward_size
        self.action_size = action_size
        
        self.constraints = constraints               # [(lower, upper),(lower, upper)...]
        
        # can be generalised to arbitrary compact & convex constraints by amending max_dist below
        # and also the projection function
        max_dist   = sum([(u-l)**reward_size for (l,u) in constraints])**(1.0/reward_size)
        tolerance  = 0.01
        self.kappa = max_dist / (2 * tolerance)**0.5
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        self.best_response_oracle = ActorCritic(in_size, action_size, network, hidden, 
                                                continuous, extra_input=True)
        
        self.lamb = np.asarray([random.random() for _ in range(reward_size)])
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
            
            if self.eps % UPDATE_EVERY_EPS == 0 and len(self.memory) > BATCH_SIZE:
                
                experience = self.memory.sample(sample_all=True)
                self.update(experience)
                
                
    def augment(self, state, kappa):
        
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
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
        projection = (x-p)/max(1, np.linalg.norm(x-p, 2))
        
        return projection
            
        
    def update(self, experiences):
 
        states, actions, rewards, next_states, dones = experiences
        
        # update best_response_oracle with normal learning
        
        rewards2 = torch.tensor([-np.dot(self.lamb, r.cpu()) for r in rewards]).reshape(-1,1).float().to(device)
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
        lamb_grad /= eps #dones.sum() # counts the episodes
        
        self.lamb = self.project(self.lamb + LAMBDA_RL_2 * lamb_grad)

##################################################

class RCPO:
    
    # Reward-Constrained Policy Optimisation
    # https://arxiv.org/pdf/1805.11074.pdf
    
    def __init__(self, action_size, constraint, in_size=4, network='DNN', hidden=16, continuous=False):
            
        self.t = 0                                   # total number of frames observed
        self.discount = 0.99                         # discount
        self.constraint = constraint
        
        self.action_size = action_size
        self.continuous = continuous
        
        self.actor = make_network('policy', network, in_size, hidden, action_size, continuous)
        self.critic = make_network('prediction', network, in_size, hidden, 1, continuous)
            
        self.lamb   = random.random()
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.LAMBDA_LR = 0.0001
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
        
    
    def act(self, state):
        if self.continuous:
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
        
        #if done:
        if self.t % BATCH_SIZE == 0:
            experiences = self.memory.sample(sample_all=True)
            
            self.update_critic(experiences)
            self.update_actor(experiences)
            self.update_lagrange(experiences)
            
            self.memory.memory.clear()
            
            
    def update_actor(self, experiences):
 
        states, actions, rewards, next_states, dones = experiences    
        rewards = (rewards[:,0] - self.lamb * rewards[:,1]).unsqueeze(-1)

        self.actor.train()
        
        with torch.no_grad():
            baseline = self.critic(states.to(device))
            outcome  = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
            advantage = (outcome - baseline).detach()
        
        if self.continuous:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-3))
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
        rewards = (rewards[:,0] - self.lamb * rewards[:,1]).reshape(-1,1)
    
        self.critic.train()
        
        prediction = self.critic(states.to(device))
        with torch.no_grad():
            target = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
            
        loss = nn.MSELoss()(prediction, target).to(device)
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        self.critic.eval()
        
        
    def update_lagrange(self, experiences):
        
        states, _, rewards, next_states, dones = experiences
        cost = sum(rewards[:,1])
        self.lamb += self.LAMBDA_LR * (cost - self.constraint)
        self.lamb = max(self.lamb, 0)

##################################################

class VaR_PG:
    
    # Policy Gradient for VaR-constrained problems
    # https://stanfordasl.github.io/wp-content/papercite-data/pdf/Chow.Ghavamzadeh.Janson.Pavone.JMLR18.pdf
    
    def __init__(self, action_size, alpha, beta, in_size=4, network='DNN', hidden=16, continuous=False):
            
        self.t = 0                                   # total number of frames observed
        self.discount = 0.99                         # discount
        
        self.alpha = alpha # constraint
        self.beta  = beta  # tolerance
        self.lamb   = random.random()
        
        self.LAMBDA_LR = 0.01
        
        self.action_size = action_size
        self.continuous = continuous
        
        self.actor = make_network('policy', network, in_size, hidden, action_size, continuous)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        if torch.cuda.is_available():
            self.actor.cuda()
        
    
    def act(self, state):
        if self.continuous:
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
        
        if self.t % BUFFER_SIZE == 0:
            experiences = self.memory.sample(sample_all=True)
            
            self.update_actor(experiences)
            self.update_lagrange(experiences)
            
            self.memory.memory.clear()
            
            
    def update_actor(self, experiences):
 
        states, actions, rewards, next_states, dones = experiences
        
        reward = sum(rewards[:,0])
        cost = 1 if sum(rewards[:,1]) > self.alpha else 0

        self.actor.train()
            
        if self.continuous:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-3))
            p2 = - torch.log(torch.sqrt(2 * math.pi * variances))
            log_probs = p1 + p2
        else:
            dists = self.actor(states.to(device))
            log_probs = torch.log(torch.gather(dists, 1, actions))
       
        loss = -(log_probs * (reward + self.lamb * cost)).mean()
        
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        self.actor.eval()
        
        
    def update_lagrange(self, experiences):
        
        states, _, rewards, next_states, dones = experiences
        cost = 1 if sum(rewards[:,1]) > self.alpha else 0
        self.lamb += self.LAMBDA_LR * (cost - self.beta)
        self.lamb = max(self.lamb, 0) 
        
##################################################

class VaR_AC:
    
    # Policy Gradient for VaR-constrained problems
    # https://stanfordasl.github.io/wp-content/papercite-data/pdf/Chow.Ghavamzadeh.Janson.Pavone.JMLR18.pdf
    
    def __init__(self, action_size, alpha, beta, in_size=4, network='DNN', hidden=16, continuous=False):
            
        self.t = 0                                   # total number of frames observed
        self.discount = 0.99                         # discount
        
        self.alpha = alpha  # constraint
        self.beta  = beta   # tolerance
        
        self.LAMBDA_LR = 0.01
        
        self.cumulative_cost = 0
        
        self.action_size = action_size  
        self.continuous = continuous
        
        self.actor = make_network('policy', network, in_size, hidden, action_size, continuous, extra_input=True)
        self.critic = make_network('prediction', network, in_size, hidden, 1, continuous, extra_input=True)
        
        self.lamb   = random.random()
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
        
    
    def act(self, state):
        
        state = self.augment(state, self.cumulative_cost)
        
        if self.continuous:
            mu, var = self.actor(state)
            mu = mu.data.cpu().numpy()
            sigma = torch.sqrt(var).data.cpu().numpy() 
            action = np.random.normal(mu, sigma)
            return torch.tensor(np.clip(action, -1, 1))
        else:
            return Categorical(self.actor(state)).sample()
        
    
    def augment(self, state, cost):
        
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
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
            outcome  = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
            advantage = (outcome - baseline).detach()
            
        if self.continuous:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-3))
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
            target = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
            
        loss = nn.MSELoss()(prediction, target).to(device)
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        self.critic.eval()
        
        
    def update_lagrange(self, experiences):
        
        states, _, rewards, next_states, dones = experiences
        #cost = 1 if sum(rewards[:,1]) > self.alpha else 07
        cost = 1 if states[-1][-1] < 0 else 0
        self.lamb += self.LAMBDA_LR * (cost - self.beta)
        self.lamb = max(self.lamb, 0)