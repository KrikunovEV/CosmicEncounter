import torch
import matplotlib.pyplot as plt
import numpy as np


path = 'three_new/'

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
for i in range(5):
    state = torch.load(path + str(i) + '.pt')
    data = state['losses']
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    j = i % 3
    i = i // 3
    ax[i][j].set_title('wins: ' + str(parties_won) + ', score = ' + '%.2f' % reward_cum[-1])
    ax[i][j].plot(np.convolve(data, np.full(31, 1. / 31), mode='valid'))
    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('loss')
fig.tight_layout()
plt.savefig(path + 'loss.png')

plt.clf()
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
for i in range(5):
    state = torch.load(path + str(i) + '.pt')
    data = state['mean_values']
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    j = i % 3
    i = i // 3
    ax[i][j].set_title('wins: ' + str(parties_won) + ', score = ' + '%.2f' % reward_cum[-1])
    ax[i][j].plot(np.convolve(data, np.full(15, 1. / 15), mode='valid'))
    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('value')
    ax[i][j].set_yticks(np.arange(0, 31, 5))
fig.tight_layout()
plt.savefig(path + 'mean_value.png')

plt.clf()
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
for i in range(5):
    state = torch.load(path + str(i) + '.pt')
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    j = i % 3
    i = i // 3
    ax[i][j].set_title('wins: ' + str(parties_won) + ', score = ' + '%.2f' % reward_cum[-1])
    ax[i][j].plot(reward_cum)
    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('cumulative reward')
    ax[i][j].set_yticks(np.arange(0, 1001, 100))
fig.tight_layout()
plt.savefig(path + 'rewards.png')

plt.figure(figsize=(16, 9))
state = torch.load(path + '0.pt')
data = np.convolve(state['episode_encounters'], np.full(31, 1. / 31), mode='valid')
plt.title('number of encounters')
plt.xlabel('episode')
plt.ylabel('amount')
plt.plot(data)
plt.tight_layout()
plt.savefig(path + 'encounters.png')
