import torch
import matplotlib.pyplot as plt
import numpy as np


path = 'nobody_2350/'
start = 0
end = 5

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
for i in range(start, end):
    state = torch.load(path + str(i) + '.pt')
    data = state['losses']
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    i -= start
    j = i % 3
    i = i // 3
    ax[i][j].set_title('wins: ' + str(parties_won) + ', score = ' + '%.2f' % reward_cum[-1])
    ax[i][j].plot(np.convolve(data, np.full(51, 1. / 51), mode='valid'))
    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('loss')
fig.tight_layout()
plt.savefig(path + str(start) + '-' + str(end) + '_loss.png')

plt.clf()
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
for i in range(start, end):
    state = torch.load(path + str(i) + '.pt')
    data = state['mean_values']
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    i -= start
    j = i % 3
    i = i // 3
    ax[i][j].set_title('wins: ' + str(parties_won) + ', score = ' + '%.2f' % reward_cum[-1])
    ax[i][j].plot(np.convolve(data, np.full(51, 1. / 51), mode='valid'))
    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('value')
    ax[i][j].set_yticks(np.arange(0, 31, 5))
fig.tight_layout()
plt.savefig(path + str(start) + '-' + str(end) + '_mean_value.png')

plt.clf()
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[1][2].axis("off")
for i in range(start, end):
    state = torch.load(path + str(i) + '.pt')
    parties_won = state['parties_won']
    reward_cum = state['reward_cum']
    i -= start
    j = i % 3
    i = i // 3
    ax[i][j].set_title('wins: ' + str(parties_won) + ', score = ' + '%.2f' % reward_cum[-1])
    ax[i][j].plot(reward_cum)
    ax[i][j].set_xlabel('episode')
    ax[i][j].set_ylabel('cumulative reward')
    ax[i][j].set_yticks(np.arange(0, 5001, 500))
fig.tight_layout()
plt.savefig(path + str(start) + '-' + str(end) + '_rewards.png')

plt.figure(figsize=(16, 9))
state = torch.load(path + '0.pt')
data = np.convolve(state['episode_encounters'], np.full(51, 1. / 51), mode='valid')
plt.title('Episode duration')
plt.xlabel('episode')
plt.ylabel('amount of encounters')
plt.plot(data)
plt.tight_layout()
plt.savefig(path + 'encounters.png')
