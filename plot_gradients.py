import matplotlib.pyplot as plt
import torch
import numpy as np


players = 5
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
path = 'everyone_grad_2350/'
for i in range(players):
    state = torch.load(path + str(i) + '.pt')
    data = state['grad_W']
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    j = i % 3
    i = i // 3
    ax[i][j].set_title('wins: ' + str(parties_won) + ', score = ' + '%.2f' % reward_cum[-1])
    ax[i][j].plot(np.convolve(data, np.full(51, 1. / 51), mode='valid'))
    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('Weight grad magnitude')
fig.tight_layout()
plt.savefig('everyone_grad_2350/weight_grad.png')

plt.clf()
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
for i in range(players):
    state = torch.load(path + str(i) + '.pt')
    data = state['grad_b']
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    j = i % 3
    i = i // 3
    ax[i][j].set_title('wins: ' + str(parties_won) + ', score = ' + '%.2f' % reward_cum[-1])
    ax[i][j].plot(np.convolve(data, np.full(51, 1. / 51), mode='valid'))
    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('Bias grad magnitude')
fig.tight_layout()
plt.savefig('everyone_grad_2350/bias_grad.png')