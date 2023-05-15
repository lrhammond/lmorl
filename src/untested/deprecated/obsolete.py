# Various obsolete functions from previous versions

class OldLexActorCritic:
    
    # lexicographic actor-critic
    
    def __init__(self, in_size, action_size, reward_size=2, network='DNN', hidden=16, is_action_cont=False, sequential=False):
            
        self.t = 0                                   # total number of frames observed
        self.eps = 0                                 # total number of episodes completed
        self.discount = 0.99                         # discount            # LH: are we keeping this constant across rewards then?
        
        self.reward_size = reward_size

        # LH: learning rates, probably need to be tweaked
        self.alpha = 0.01
        self.beta = [0.001 + 0.001*((r + 1) / reward_size) for r in range(reward_size)]
        self.eta = [0.0001 + 0.0001*((r + 1) / reward_size) for r in range(reward_size)]
        
        # LH: for storing quantities for the objective functions and their gradients
        self.j = [0 for _ in range(reward_size)]
        #self.j_grad = []
        
        # LH: solve constrained problems sequentially or synchronously (defaults to false)
        self.sequential = sequential
        self.is_action_cont = is_action_cont
        
        # LH: not totally sure how to extract this from the environment... (TODO), should be an 1 x |S| probability vector
        # self.initial_dist = initial_dist
        
        # LH: defines tolerance to check convergence and number of previous losses to check for convergence
        self.tol = TOL      # LH: TODO based on experiment, should be on the order of expected std of lagrangian once converged
        self.prev = 50      # LH: not sure this is a good number
        
        # LH: for sequential use, stores previous losses/gradients to check convergence which objective we are optimising
        self.losses = [collections.deque(maxlen=self.prev) for r in range(reward_size)]
        self.grads = [collections.deque(maxlen=self.prev) for r in range(reward_size)]
        self.current = 0
        
        self.actor = make_network('policy', network, in_size, hidden, action_size, is_action_cont)
        self.critic = make_network('prediction', network, in_size, hidden, reward_size, is_action_cont)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()

        # LH: initialised multipliers to 0 (not sure how much difference it makes but it's what I've seen done in, e.g., the RCPO paper)
        self.mu = [0.0 for r in range(reward_size)]

        # LH: don't need lambdas with relu nets
        ##### self.lamb = [torch.autograd.grad(0, self.actor.parameters()) for r in reward_size]
        
        # model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # self.lamb = [torch.zeros(params) for r in range(reward_size)]

    # LH: checks if policy has converged w.r.t. losses from critic r
    def converged(self, r):

        # LH: for previous critics
        if r < self.current:
            return True
        # LH: for current critic
        losses = np.array(self.losses[r])
        if np.shape(losses)[0] < self.prev or np.std([float(i) for i in losses]) < self.tol:
            return False 
        else:
            return True

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
        self.memory.add(state, action, torch.tensor(rewards), next_state, done)
        
        if done:
            self.eps += 1

            # LH: added this for solving lexicographic problems in sequence rather than synchronously (defaults to false)
            if self.sequential:

                r = self.current
                self.update_critic(self.memory.sample(sample_all=True))
                self.update_actor(self.memory.sample(sample_all=True), r)
                
                for k in range(r-1):
                    self.update_lagrange(self.memory.sample(sample_all=True), k)

                # LH: if converged store values for this objective and its gradient
                if self.converged(r):
                    
                    ##### self.j.append(self.initial_dist * self.critic(states.to(device))[:,r])
                    self.j.append(self.critic(states[0].unsqueeze(0).to(device))[:,r])
                    
                    #self.j_grad.append(np.mean(np.array(self.grads[r]), axis=0))
                    self.current += 1

            # LH: this is the regular synchronous version from the pseudocode in the paper, but (theoretical) convergence depends on the learning rates
            else:

                #if len(self.j) == 0:
                    ##### self.j.append(self.initial_dist * self.critic(states.to(device))[:,r])
                #    self.j.append(self.critic(states[0].unsqueeze(0).to(device))[:,r])
                #    self.j_grad.append(np.mean(np.array(self.grads[r]), axis=0))
                
                # LH: perform updates
                self.update_critic(self.memory.sample(sample_all=True))
                
                for r in range(self.reward_size):
                    self.update_actor(self.memory.sample(sample_all=True), r)
                    # # LH: no constants to update with when t = 0
                    # if self.t != 0:
                    self.update_lagrange(self.memory.sample(sample_all=True), r)

                    # LH: if not converged keep updating stored objective and gradient values, freeze (in order) once converged
                    if not self.converged(r):  
                        ##### self.j[r] = self.initial_dist * self.critic(states.to(device))[:,r])
                        states,_,_,_,_ = self.memory.sample(sample_all=True)
                        self.j[r] = self.critic(states[0].unsqueeze(0).to(device))[:,r]
                        # self.j_grad[r] = np.mean(np.array(self.grads[r]), axis=0)
                        
                    elif r == self.current:
                        self.current += 1
                    
            self.memory.memory.clear()
            

    # LH: changed to update based on one critic at a time
    def update_actor(self, experiences, r):

        # LH: set learning rate
        #for param_group in self.actor_optimizer.param_groups:
        #    param_group['lr'] = self.beta[r]
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.beta[r])
        
        states, actions, rewards, next_states, dones = experiences
        
        self.actor.train()

        with torch.no_grad():
            baseline = self.critic(states.to(device))[:,r].unsqueeze(-1)
            outcome  = rewards[:,r].unsqueeze(-1) + (self.discount * self.critic(next_states.to(device))[:,r].unsqueeze(-1) * (1-dones))
            advantage = (outcome - baseline).detach()        

        if self.is_action_cont:
            means, variances = self.actor(states.to(device))
            p1 = - ((means - actions) ** 2) / (2 * variances.clamp(min=1e-3))
            p2 = - torch.log(torch.sqrt(2 * math.pi * variances))
            log_probs = p1 + p2
        else:
            dists = self.actor(states.to(device))
            log_probs = torch.log(torch.gather(dists, 1, actions))

        # LH: for lambdas, not needed
        # params = torch.cat([param.view(-1) for param in self.actor.parameters()])
        # h = hessian(probs.squeeze(), params)

        #for i in range(h.shape[0]):
        #    print(h[i,:,:])
        
        # LH: add primary objective term
        sum_loss = log_probs * advantage

        #loss = -(log_probs * advantage).mean()

        # LH: for lambdas, not needed
        # second_order_term = np.dot(0, advantage)

        # LH: get gradient and record
        #grad = torch.autograd.grad(-(sum_loss).mean(), self.actor.parameters(), retain_graph=True)
        #self.grads[r].append(grad)

        for k in range(r):
            with torch.no_grad():
                baseline = self.critic(states.to(device))[:,k].unsqueeze(-1)
                outcome  = rewards[:,k].unsqueeze(-1) + (self.discount * self.critic(next_states.to(device))[:,k].unsqueeze(-1) * (1-dones))
                advantage = (outcome - baseline).detach()
                # LH: add mu terms
                sum_loss += self.mu[k] * log_probs * advantage

                # LH: for lambdas, not needed
                # second_order_term += advantage * torch.mm(h, self.lamb[k])

        # LH: define overall loss for Lagrangian

        loss = -(sum_loss).mean()
        self.losses[r].append(loss)
        
        self.actor_optimizer.zero_grad()
        loss.backward()

        self.actor_optimizer.step()
        self.actor.eval()
        

    def update_critic(self, experiences):

        # the critic is updated in the normal way

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.alpha)
        
        states, _, rewards, next_states, dones = experiences
    
        self.critic.train()
        
        prediction = self.critic(states.to(device))
        with torch.no_grad():
            target = rewards + (self.discount * self.critic(next_states.to(device)) * (1-dones))
        
        loss = nn.MSELoss()(prediction, target).to(device)
        
        self.critic_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        self.critic.eval()


    def update_lagrange(self, experiences, r):

        states, _, _, _, _ = experiences
        
        current_j_r = self.critic(states[0].unsqueeze(0).to(device))[:,r]
        self.mu[r] += self.eta[r] * (self.j[r] - current_j_r)
        self.mu[r] = max(self.mu[r], 0)

##################################################

class OldOldLexActorCritic:
    
    # lexicographic actor-critic
    
    def __init__(self, in_size, action_size, reward_size=2, network='DNN', hidden=16):
            
        self.t = 0                                   # total number of frames observed
        self.discount = 0.99                         # discount
        
        self.reward_size = reward_size
        
        if network=='DNN':
            self.actor  = PolicyDNN(in_size, action_size, hidden)
            self.critic = DNN(in_size, reward_size, hidden)            
        elif network=='CNN':
            self.actor = PolicyCNN(int((in_size/3)**0.5), channels=3, 
                                   convs=hidden, action_size=action_size, hidden=hidden)
            self.critic = CNN(int((in_size/3)**0.5), channels=3, 
                              out_size=reward_size, 
                              convs=hidden, hidden=hidden)
        else:
            print('invalid network specification')
            assert False

        
        self.memory = ReplayBuffer(BATCH_SIZE, BATCH_SIZE)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())
        
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
        
        self.losses = [collections.deque(maxlen=CONVERGENCE_LENGTH) for _ in range(reward_size)]
    
        self.converged = []
    
    
    def act(self, state):
        return Categorical(self.actor(state)).sample()
    
    
    def step(self, state, action, rewards, next_state, done):
        
        self.t += 1
        self.memory.add(state, action, torch.tensor(rewards), next_state, done)
        
        #if done:
        if self.t % BATCH_SIZE == 0:
            self.update_critic(self.memory.sample(sample_all=True))
            self.update_actor(self.memory.sample(sample_all=True))
            self.memory.memory.clear()
            
            
    def update_actor(self, experiences):
         
        for r in range(self.reward_size):
            if not self.has_converged(r):
                self.update_actor_for_reward(experiences, r)
                return
        self.update_actor_for_reward(experiences, 0)
        
        
    def has_converged(self, reward):
        
        sequence = self.losses[reward]
        
        if len(sequence) < CONVERGENCE_LENGTH:
            return False
        
        if max(sequence)-min(sequence) > CONVERGENCE_DEVIATION:
            return False
        else:
            if not reward in self.converged:
                self.converged.append(reward)
                print('Reward {} has converged!'.format(reward))
            return True
    
    
    def update_actor_for_reward(self, experiences, r):
    
        states, actions, rewards, next_states, dones = experiences
        self.actor.train()
        
        with torch.no_grad():
            baseline = self.critic(states.to(device))[:,r]
            outcome  = rewards[:,r] + (self.discount * self.critic(next_states.to(device))[:,r] * (1-dones))
            advantage = (outcome - baseline).detach()
            
        dists = self.actor(states.to(device))
        log_probs = torch.log(torch.gather(dists, 1, actions))
    
        loss = -(log_probs * advantage).mean()
        self.losses[r].append(float(loss))
        
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
    
##################################################

class LexNaturalActorCritic:
    
    def __init__(self, in_size, action_size, reward_size=2, network='DNN', hidden=16, sequential=False):
            
        self.t = 0                                   # total number of frames observed
        self.eps = 0                                 # total number of episodes completed
        self.discount = 0.99                         # discount            # LH: are we keeping this constant across rewards then?
        
        self.reward_size = reward_size

        # LH: learning rates, probably need to be tweaked
        self.alpha = 0.1
        self.beta = [0.01 + 0.01*((r + 1) / reward_size) for r in range(reward_size)]
        self.eta = [0.001 + 0.001*((r + 1) / reward_size) for r in range(reward_size)]
        self.iota = 0.1
        # LH: for storing quantities for the loss functions and their gradients
        self.l = [0.0 for r in range(reward_size)]
        # LH: don't need when lambdas = 0
        # self.l_grad = []
        # LH: solve constrained problems sequentially or synchronously (defaults to false)
        self.sequential = sequential
        # LH: not totally sure how to extract this from the environment... (TODO), should be a 1 x |S| probability vector
        ##### self.initial_dist = initial_dist
        # LH: defines tolerance to check convergence and number of previous losses to check for convergence
        self.tol = TOL      # LH: TODO based on experiment, should be on the order of expected std of lagrangian once converged
        self.tol2 = TOL2    # LH: TODO based on experiment, should be on the order of the mean std (i.e. elementwise std of a set of vectors, then the mean of this) of natural gradient once converged
        self.prev = 10      # LH: not sure this is a good number
        # LH: for sequential use, stores previous losses/gradients to check convergence which objective we are optimising
        self.individual_losses = [collections.deque(maxlen=self.prev) for r in range(reward_size)]
        self.losses = [collections.deque(maxlen=self.prev) for r in range(reward_size)]
        # LH: don't need when lambdas = 0
        # self.individual_grads = [collections.deque(maxlen=self.prev) for r in range(reward_size)]
        self.nat_grads = collections.deque(maxlen=self.prev)
        self.current = 0

        if network=='DNN':
            self.actor  = PolicyDNN(in_size, action_size, hidden)     
            # # LH: here we use a linear critic
            # self.critic = nn.Linear(feature_size, reward_size, bias=False)
            self.critic = DNN(in_size, reward_size, hidden)
        elif network=='CNN':
            self.actor = PolicyCNN(int((in_size/3)**0.5), channels=3, 
                                   convs=hidden, action_size=action_size, hidden=hidden)
            # # LH: here we use a linear critic
            # self.critic = nn.Linear(feature_size, reward_size, bias=False)
            self.critic = CNN(int((in_size/3)**0.5), channels=3, out_size=reward_size, convs=hidden, hidden=hidden)
        else:
            print('invalid network specification')
            assert False

        self.memory = ReplayBuffer(int(1e6), 32)
        
        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()

        # LH: initialise natural gradient
        model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
        #params = sum([np.prod(p.size()) for p in model_parameters])
        # LH: TODO may need to be slightly reshaped somehow
        #self.x = torch.zeros(params)
        self.x = [torch.zeros(p.shape) for p in model_parameters]

        # LH: initialised multipliers to 0 (not sure how much difference it makes but it's what I've seen done in, e.g., the RCPO paper)
        self.mu = [0.0 for r in range(reward_size)]
        # LH: lambdas not necessary for MAE losses
        # self.lamb = [torch.autograd.grad(0, self.actor.parameters()) for r in reward_size]
    

    # LH: checks if nat_grad has converged w.r.t. losses from critic r
    def converged(self, r):

        # # LH: for nat_grad overall
        # if overall:
        #     xs = np.array(self.nat_grads)
        #     if np.mean(np.std(xs, axis=0)) < self.tol2:
        #         return True
        #     else:
        #         return False

        # # LH: w.r.t. a certain critic 
        # else:

        # LH: for previous critics
        if r < self.current:
            return True
        # LH: for current critic
        losses = np.array(self.losses[r])
        if np.shape(losses)[0] < self.prev or np.std([float(i) for i in losses]) < self.tol:
            return False
        else:
            return True


    # LH: L-NAC needs a linear critic, so this is a function from states to some feature_dim-dimensional vector, TODO
    def features(self, states):

        # LH: currently just returns states so that the linear critic can be swapped out for the DNN one without changing anything below, can be changed by uncommenting the relevant parts about how the critic is created above and by adding a feature_size input to the arguments of the class initialiser
        return states 


    # LH: variable learning rates, not currently used

    # # LH: learning rate for critic i at timestep t
    # def alpha(self, i, t):

    # # LH: learning rate for actor i at timestep t
    # def beta(self, i, t):

    # # LH: learning rate for langrange multipliers i at timestep t
    # def eta(self, i, t):


    def act(self, state):
        return Categorical(self.actor(state)).sample()


    def step(self, state, action, rewards, next_state, done):
        
        self.t += 1
        self.memory.add(state, action, torch.tensor(rewards), next_state, done)
        
        if done:
            # if the episode is finished, perform an update

            self.eps += 1

            score = self.compute_scores(self.memory.sample(sample_all=True))

            # LH: added this for solving lexicographic problems in sequence rather than synchronously (defaults to false)
            if self.sequential:

                r = self.current
                self.update_critic(self.memory.sample(sample_all=True))
                self.update_nat_grad(self.memory.sample(sample_all=True), score, r)
                for k in range(r-1):
                    self.update_lagrange(self.memory.sample(sample_all=True), score, k)

                # LH: if converged store values for this objective and its gradient
                if self.converged(r):
                    self.l[r] = np.mean(np.array(self.individual_losses[r]))
                    # LH: don't need when lambdas = 0
                    # self.l_grad.append(np.mean(np.array(self.individual_grads[r]), axis=0))
                    if self.current == self.reward_size-1:
                        self.update_actor()
                        self.current = 0
                        self.l = [0.0 for r in range(reward_size)]
                        # self.l_grad = []
                    else:
                        self.current += 1  

            # LH: this is the regular synchronous version from the pseudocode in the paper, but (theoretical) convergence depends on the learning rates
            else:
                
                # LH: perform updates
                self.update_critic(self.memory.sample(sample_all=True))
                for r in range(self.reward_size):
                    self.update_nat_grad(self.memory.sample(sample_all=True), score, r)
                    self.update_lagrange(self.memory.sample(sample_all=True), score, r)

                    # LH: if not converged keep updating stored objective and gradient values, freeze (in order) once converged
                    if not self.converged(r):
                        # if t == 0:
                        #     self.l.append(np.mean(np.array(self.individual_losses[r])))
                        #     self.l_grad.append(np.mean(np.array(self.individual_grads[r]), axis=0))
                        # else:
                        self.l[r] = np.mean(np.array(self.individual_losses[r]))
                        # LH: don't need when lambdas = 0
                        # self.l_grad[r] = np.mean(np.array(self.individual_grads[r]), axis=0)
                    elif r == self.current:
                        if self.current == self.reward_size-1:
                            self.update_actor()
                            self.current = 0
                            self.l = [0.0 for r in range(self.reward_size)]
                            # self.l_grad = []
                        else:
                            self.current += 1  

                # # LH: update actor anyway to speed things up
                # self.update_actor()

            self.memory.memory.clear()
            
            
            # LH: update actor if nat_grad converged and reset various stored quantities for a new round of computing nat_grad
            # if self.converged(0, overall=True):
            #     self.update_actor()
            #     self.current = 0
            #     self.l = [0.0 for r in range(reward_size)]
            #     # self.l_grad = []



    def update_actor(self):
        # LH: set learning rate
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.iota)
        # LH: manually update weights using natural gradient x
        self.actor.train()
        self.actor_optimizer.zero_grad()
        # LH: this is ugly, sorry

        #zero_loss = torch.zeros(1, requires_grad=True)
        #zero_loss.backward()

        for p, xi in zip(self.actor.parameters(), self.x):
            p.grad = Variable(p.data.new(p.size()).zero_())
            p.grad = xi            

##        for i, p in enumerate(self.actor.parameters()):
##            if p.requires_grad:
##
##                p.grad = Variable(p.data.new(p.size()).zero_())
##                
##                print(p.grad)
##                print(self.x[i].shape)
##                print(self.x[i])
##                
##                p.grad = self.x[i]
                
        self.actor_optimizer.step()
        self.actor.eval()
        

    def update_critic(self, experiences):

        # the critic is updated in the normal way

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.alpha)
        
        states, _, rewards, next_states, dones = experiences
    
        self.critic.train()
        
        prediction = self.critic(self.features(states.to(device))) # predicted outcomes
        with torch.no_grad():
            target = rewards + (self.discount * self.critic(self.features(next_states.to(device))) * (1-dones)) # rewards + predicted subsequent outcomes
        
        # LH: left this same for now but not sure whether we should use the classical TD semi-gradient update instead, which I don't think this is equivalent to?
        loss = nn.MSELoss()(prediction, target).to(device) # loss = MSE of TD error
        
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        self.critic.eval()

    # LH: hopefully factoring this out will boost efficiency slightly
    def compute_scores(self, experiences):

        states, actions, _, _, _ = experiences
        dists = self.actor(states.to(device))
        probs = torch.gather(dists, 1, actions)
        score = [torch.autograd.grad(p, self.actor.parameters(), retain_graph=True) for p in torch.log(probs)]
        #score = [np.concatenate([i.flatten() for i in score[j]]).reshape((-1,1)) for j in range(len(score))]        
        #score = torch.tensor(np.concatenate(score, axis=1))

        return score


    def update_nat_grad(self, experiences, score, r):

        states, actions, rewards, next_states, dones = experiences

        # dists = self.actor(states.to(device))
        # probs = torch.gather(dists, 1, actions)
        # # score = torch.autograd.grad(torch.log(probs), self.actor.parameters())
        
        # score = [torch.autograd.grad(p, self.actor.parameters(), retain_graph=True) for p in torch.log(probs)]
        # score = [np.concatenate([i.flatten() for i in score[j]]).reshape((-1,1)) for j in range(len(score))]
        # score = torch.tensor(np.concatenate(score, axis=1))

        with torch.no_grad():
            #baseline = self.critic(self.features(states.to(device)))[:,r]
            #outcome  = rewards[:,r] + (self.discount * self.critic(self.features(next_states.to(device)))[:,r] * (1-dones))
            baseline = self.critic(states.to(device))[:,r]
            outcome  = rewards[:,r] + (self.discount * self.critic(next_states.to(device))[:,r] * (1-dones.squeeze()))
            advantage = (outcome - baseline).detach()

        # LH: might need to reshape score here but I'm not exactly sure what to, this is meant to be a vector of (x^\top score(s,a) - A(s,a)) for different s,a (TODO)
            
        x = np.concatenate([i.flatten().unsqueeze(0) for i in self.x], axis=1)
        x = torch.tensor(x)

        sc = [np.concatenate([i.flatten().unsqueeze(0) for i in p], axis=1) for p in score]
        sc = torch.tensor(np.concatenate(sc, axis=0)).transpose(0,1)
        
        l = torch.mm(x, sc) - advantage
        
        ls = []
        for k in range(0, r):
            with torch.no_grad():
                baseline = self.critic(states.to(device))[:,k]
                outcome  = rewards[:,k] + (self.discount * self.critic(next_states.to(device))[:,k] * (1-dones.squeeze()))
                advantage = (outcome - baseline).detach()
                # LH: same here about reshaping score (TODO)
                #ls.append(np.dot(self.x, score) - advantage)

                x = np.concatenate([i.flatten().unsqueeze(0) for i in self.x], axis=1)
                x = torch.tensor(x)

                sc = [np.concatenate([i.flatten().unsqueeze(0) for i in p], axis=1) for p in score]
                sc = torch.tensor(np.concatenate(sc, axis=0)).transpose(0,1)
                
                ls.append(torch.mm(x, sc) - advantage)

        # LH: record overall Lagrangian value from this episode
        loss = np.abs(l)
        self.individual_losses[r].append(loss.mean())
        for k in range(0, r):
            
            loss += self.mu[k] * (torch.abs(ls[k]) - torch.tensor(self.l[k] * np.ones(advantage.shape)))

        # LH: ignore lambda term (see paper for reasons)
        self.losses[r].append(loss.mean())

        # LH: form and save gradient
        grad = np.sign(l).mean()
        # self.individual_grads[r].append(grad)
        for k in range(0, r):
            grad += self.mu[k] * np.sign(ls[k]).mean()
            # LH: again, no lambda term
        
        # LH: update natural gradient estimate, no projection used though could add this
        
        x += torch.sum(self.beta[r] * (-grad) * sc, axis=1)

        self.nat_grads.append(torch.tensor(x))
        

    def update_lagrange(self, experiences, score, r):

        states, actions, rewards, next_states, dones = experiences

        # dists = self.actor(states.to(device))
        # probs = torch.gather(dists, 1, actions)
        # #score = torch.autograd.grad(torch.log(probs), self.actor.parameters())

        # score = [torch.autograd.grad(p, self.actor.parameters(), retain_graph=True) for p in torch.log(probs)]
        # score = [np.concatenate([i.flatten() for i in score[j]]).reshape((-1,1)) for j in range(len(score))]
        # score = torch.tensor(np.concatenate(score, axis=1))

        with torch.no_grad():
            baseline = self.critic(self.features(states.to(device)))[:,r]
            outcome  = rewards[:,r] + (self.discount * self.critic(self.features(next_states.to(device)))[:,r] * (1-dones))
            advantage = (outcome - baseline).detach()

        # LH: again, might need to reshape score here (TODO)

        x = np.concatenate([i.flatten().unsqueeze(0) for i in self.x], axis=1)

        sc = [np.concatenate([i.flatten().unsqueeze(0) for i in p], axis=1) for p in score]
        sc = torch.tensor(np.concatenate(sc, axis=0)).transpose(0,1)
        
        loss = torch.abs(torch.mm(torch.tensor(x), sc) - advantage).mean()
        #loss = (np.abs(np.dot(self.x, score) - advantage)).mean()
        
        # LH: update mus, no lambda updates required due to use of MAE
        self.mu[r] += self.eta[r] * (loss - self.l[r])
        self.mu[r] = max(0, self.mu[r])

##################################################

# From: https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
def jacobian(y, x, create_graph=False):
    action_size=4
    #y = torch.ones(action_size) * y  
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True,
                                     create_graph=create_graph, allow_unused=True)[0]
        if grad_x == None:
            grad_x = torch.zeros_like(x, requires_grad=True)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)

def hessian(y, x):
    #z = y**1
    return jacobian(jacobian(y, x, create_graph=True), x, create_graph=True)