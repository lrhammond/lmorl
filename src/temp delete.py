import gym_safety
import gym

env = gym.make('CartSafe-v0')

print(env.observation_space.high)
print(env.observation_space.low)
print()

for _ in range(10):
    s,r,d,i = env.reset()
    for j in range(1000):
        env.render()
        s,r,d,i = env.step(env.action_space.sample())
        cp = s[0]
        if d or abs(cp) >= 2.39:
            print(d)
            print(cp)
            print(j)
            print()
            break

env.close()
