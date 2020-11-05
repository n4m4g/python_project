import numpy as np
import matplotlib.pyplot as plt

REWARD = np.ones(100)
GAMMA = 0.95

def func1():
    discount_reward = np.zeros_like(REWARD)
    tmp = 0

    """
    Gt[-1] = R[-1]
    Gt[-2] = R[-2] + GAMMA*R[-1] = R[-2] + GAMMA*Gt[-1]
    Gt[-3] = R[-3] + GAMMA*R[-2] + GAMMA**2*R[-1] = R[-3] + GAMMA*Gt[-2]
    """
    for idx in reversed(range(len(REWARD))):
        tmp = tmp*GAMMA + REWARD[idx]
        discount_reward[idx] = tmp

    return discount_reward

def func2():
    discount_reward = np.zeros_like(REWARD)

    for idx, t in enumerate(REWARD):
        Gt = 0
        pw = 0
        for r in REWARD[idx:]:
            Gt += GAMMA**pw*r
            pw += 1
        discount_reward[idx] = Gt

    return discount_reward

print(func1())
print(func2())


