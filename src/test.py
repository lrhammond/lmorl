# Test file for running on ARC (without MuJoCo)

from learners_lexicographic import *
import os

os.makedirs('./results', exist_ok=True)

if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")

class OrbGame:
    
    def __init__(self, grid_size=10, greens=8, reds=8):
        
        assert type(greens) == int and type(reds) == int
        assert greens >= 0 and reds >= 0
        
        self.action_space = [0,1,2,3,4]
        self.grid_size = grid_size
        
        #self.grid = np.asarray([[[0,0,0] for _ in range(grid_size)] for _ in range(grid_size)])
        
        self.greens = greens
        self.reds = reds
        
        self.init_grid()
                
    def init_grid(self):
        
        self.grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=int)
        
        # place the agent
        
        self.x = random.randint(0, self.grid_size-1)
        self.y = random.randint(0, self.grid_size-1)
        
        self.grid[self.x, self.y, 0] = 1
        
        # place the green orbs
        
        i = self.greens
        while i > 0:
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            if np.count_nonzero(self.grid[x,y,:]) == 0:
                self.grid[x,y,1] = 1
                i -= 1
                
        # place the red orbs
                
        i = self.reds
        while i > 0:
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            if np.count_nonzero(self.grid[x,y,:]) == 0:
                self.grid[x,y,2] = 1
                i -= 1
                
        self.start_grid = np.array(self.grid)     

    def reset(self):
        self.grid = np.array(self.start_grid)
        self.total_greens = self.greens
        return self.preprocess(self.grid)
    
    def preprocess(self, grid):
        # batch, dim1, dim2, features
        
        state = np.reshape(grid, [1, self.grid_size, self.grid_size, 3])
        state = np.moveaxis(state,3,1) # channels must be in the 2nd axis for the CNN
        state = np.reshape(state, [1, self.grid_size*self.grid_size*3])
        state = torch.tensor(state).float().to(device)
        return state       
        
    def jiggle(self):
        
        red_collisions = 0
        green_collisions = 0
        
        # jiggle the reds
        
        exceptions = []
        
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                if self.grid[x,y,2] == 1 and (x,y) not in exceptions:
                    i,j = random.choice([(0,0),(-1,0),(1,0),(0,-1),(0,1)])
                    if (x!=0 or i!=-1) and (y!=0 or j!=-1) and (x!=self.grid_size-1 or i!=1) and (y!=self.grid_size-1 or j!=1):
                        if np.count_nonzero(self.grid[x+i,y+j,:]) == 0:
                            self.grid[x,y,2] = 0
                            self.grid[x+i, y+j, 2] = 1
                            exceptions.append((x+i, y+j))
                        elif self.grid[x+i, y+j, 0] == 1:
                            self.grid[x,y,2] = 0
                            red_collisions += 1
        
        # jiggle the greens
        
        exceptions = []
        
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                if self.grid[x,y,1] == 1 and (x,y) not in exceptions:
                    i,j = random.choice([(0,0),(-1,0),(1,0),(0,-1),(0,1)])
                    if (x!=0 or i!=-1) and (y!=0 or j!=-1) and (x!=self.grid_size-1 or i!=1) and (y!=self.grid_size-1 or j!=1):
                        if np.count_nonzero(self.grid[x+i,y+j,:]) == 0:
                            self.grid[x,y,1] = 0
                            self.grid[x+i,y+j,1] = 1
                            exceptions.append((x+i, y+j))
                        elif self.grid[x+i, y+j, 0] == 1:
                            self.grid[x,y,1] = 0
                            green_collisions += 1
                    
        return green_collisions, red_collisions      
        
    def step(self, action):
        
        # 0 = left, 1 = up, 2 = right, 3 = down, 4 = pass
        
        assert action in [0,1,2,3,4]
        
        if action == 0 and self.x != 0:
            
            self.grid[self.x, self.y, 0] = 0
            self.x -= 1
            self.grid[self.x, self.y, 0] = 1
            
        elif action == 1 and self.y != self.grid_size-1:
                
            self.grid[self.x, self.y, 0] = 0
            self.y += 1
            self.grid[self.x, self.y, 0] = 1
        
        elif action == 2 and self.x != self.grid_size-1:
                
            self.grid[self.x, self.y, 0] = 0
            self.x += 1
            self.grid[self.x, self.y, 0] = 1
        
        elif action == 3 and self.y != 0:
                
            self.grid[self.x, self.y, 0] = 0
            self.y -= 1
            self.grid[self.x, self.y, 0] = 1
            
        elif action == 4:
            pass

        # g = greens hit, r = reds hit
        
        if self.grid[self.x, self.y, 1] == 1:
            self.grid[self.x, self.y, 1] = 0
            g = 1
            r = 0
        elif self.grid[self.x, self.y, 2] == 1:
            self.grid[self.x, self.y, 2] = 0
            g = 0
            r = 1
        else:
            g = 0
            r = 0

        #gc, rc = self.jiggle()
        #g += gc
        #r += rc
        
        self.total_greens -= g
        
        # the episode is over when there are no more greens
        done = (self.total_greens == 0)
        
        # if a red is hit, the episode ends with probability 1
        #if r > 0: # and np.random.choice([True,False], p=[0.3, 0.7]):
        #    done = True
            
        # there is a certain constant probability that the episode will end
        # if a new square is entered each time step then all squares can be entered wp 0.5
        # th = 2**(-1/(self.grid_size**2))
        th = 0.01
        if np.random.choice([True,False], p=[1-th, th]):
            done = True
            
        state = self.preprocess(self.grid)
        
        return state, [g, r], done, None
    
    def render(self):
        s = ''
        for y in range(0, self.grid_size): #range(self.grid_size-1, -1, -1):
            for x in range(0, self.grid_size):
                if int(self.grid[x,y,0]) == 1:
                    s += 'A'
                elif int(self.grid[x,y,1]) == 1:
                    s += 'G'
                elif int(self.grid[x,y,2]) == 1:
                    s += 'R'
                else:
                    s += '.'
            s += '   '
            for x in range(0, self.grid_size):
                if int(self.grid[x,y,0]) == 1:
                    s += 'X'
                else:
                    s += '.'
            s += '   '
            for x in range(0, self.grid_size):
                if int(self.grid[x,y,1]) == 1:
                    s += 'X'
                else:
                    s += '.'
            s += '   '
            for x in range(0, self.grid_size):
                if int(self.grid[x,y,2]) == 1:
                    s += 'X'
                else:
                    s += '.'
            
            s += '\n'
        print(s)
        print('greens: {}'.format(self.total_greens))
        print()

        print('#################################################')
        print()
    
    def close(self):
        pass

grid_size = 5
greens = grid_size
reds   = 2*grid_size
env = OrbGame(grid_size=grid_size, greens=greens, reds=reds)

in_size = grid_size*grid_size*3
action_size=4
hidden = 8

mode = 3

episodes = 10000

agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False)

mean_score = []
mean_bombs = []

with open('./results/test.txt', 'w') as f:
    f.write('TEST\n')

for i in range(episodes+1):

    state = env.reset()
    done = False

    score = 0
    bombs = 0

    #env.render()
    while not done:

        action = agent.act(state)

        #print('ACTION: {}'.format(action))
        #print('#################################################')
        
        next_state, rewards, done, _ = env.step(action)
        
        if mode==1:
            r = rewards[0]
        elif mode==2:
            r = rewards[0]-rewards[1]
        elif mode==3:
            r = [-rewards[1], rewards[0]]
        elif mode==4:
            r = rewards
        elif mode==5:
            r = [rewards[0], -rewards[1]]
            
        agent.step(state, action, r, next_state, done)
        state = next_state

        score += rewards[0]
        bombs += rewards[1]

        #env.render()

    #print('#################################################')

    with open('./results/test.txt', 'w') as f:
        f.write('{},{}\n'.format(score,bombs))

    mean_score.append(float(score))
    mean_bombs.append(float(bombs))

    if i % 1000 == 0 and i > 0:
        # print("Episodes: {}    Mean Score: {}    Mean Bombs: {}".format(i, torch.tensor(mean_score)[-1000:].mean(), torch.tensor(mean_bombs)[-1000:].mean()))
        print("Episodes: {}    Mean Score: {}    Mean Bombs: {}".format(i, torch.tensor(mean_score).mean(), torch.tensor(mean_bombs).mean()))
