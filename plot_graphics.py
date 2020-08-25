import torch
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import numpy as np


fig, ax = plt.subplots(2, 5, figsize=(16, 9))

for i in range(5):
    state = torch.load('three_new/' + str(i) + '.pt')
    losses = state['losses']
    parties_won = state['parties_won']
    ax[0][i].set_title('wins: ' + str(parties_won))
    ax[0][i].plot(median_filter(losses, 10))
    ax[0][i].set_xlabel('episode')
    ax[0][i].set_ylabel('loss')

for i in range(5):
    state = torch.load('three_new/' + str(i) + '.pt')
    values = state['mean_values']
    ax[1][i].plot(median_filter(values, 3))
    ax[1][i].set_xlabel('episode')
    ax[1][i].set_ylabel('mean value')
    ax[1][i].set_yticks(np.arange(0, 31, 5))
fig.tight_layout()

plt.savefig('two.png')
plt.show()

