#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

vel, dist = np.meshgrid(np.arange(0, 1, 0.01),
                        np.arange(0, 1, 0.01))

# reward = 1 - dist**0.4
dist_reward = 1 - dist**0.4

tmp1 = np.zeros_like(vel)
tmp1.fill(0.1)
tmp2 = np.zeros_like(dist)
tmp2.fill(0.1)

base = 1 - np.maximum(tmp1, vel)
power = 1 / np.maximum(tmp2, dist)

vel_discount = base**power
reward = vel_discount * dist_reward

print(vel.shape)
print(dist.shape)
print(reward.shape)
l_v, h_v = vel.min(), vel.max()
l_d, h_d = dist.min(), dist.max()
l_r, h_r = -np.abs(reward).max(), np.abs(reward).max()

fig, axes = plt.subplots()

c = axes.pcolormesh(dist, vel, reward,
                    cmap='coolwarm',
                    vmin=l_r,
                    vmax=h_r,
                    shading='auto')

axes.set_title('Heatmap')
axes.axis([l_d, h_d, l_v, h_v])
fig.colorbar(c)

plt.show()
