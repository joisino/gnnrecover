import os
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.sparse import csr_matrix
import networkx as nx

from scipy.linalg import orthogonal_procrustes

import torch
import torch.optim as optim
from torch_geometric.data import Data

from sklearn.manifold import TSNE

from util import Net, GIN, GAT, moon, stationary, reconstruct, dG


np.random.seed(0)
torch.manual_seed(0)

n = 5000
m = 500
x, n = moon(n)
n_train = int(n * 0.7)
train_ind = torch.randperm(n)[:n_train]
test_ind = torch.LongTensor(list(set(np.arange(n)) - set(train_ind.tolist())))
K = int(np.sqrt(n) * np.log2(n) / 10)
D = pairwise_distances(x)
fr = np.arange(n).repeat(K).reshape(-1)
to = np.argsort(D, axis=1)[:, 1:K + 1].reshape(-1)
A = csr_matrix((np.ones(n * K) / K, (fr, to)))

edge_index = np.vstack([fr, to])
edge_index = torch.tensor(edge_index, dtype=torch.long)
X = torch.tensor([[K, n] for i in range(n)], dtype=torch.float)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for i in range(100):
    # Note 1: In the original formulation, $g$, i.e., the neural network for the scale function, should be used in reconstruct(K, pr, n, m, fr, to), namely, in the definition of $s$. We factorize $s$ and multiply g after we reconstruct the features. This is mathematically equivalent. We do this to avoid memory overflow due to long backpropagation.
    # Note 2: We roughly standardize n for stability by (n - 3000) / 3000. This does not affect the representational power of GNNs by merging them into the network parameters.
    pr = stationary(A)
    pr = np.maximum(pr, 1e-9)
    rec_orig = reconstruct(K, pr, n, m, fr, to)
    rec_orig = torch.FloatTensor(rec_orig)
    g = net(torch.FloatTensor([(n - 3000) / 3000]))
    rec = rec_orig * (g ** 0.5)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])

    print(n, float(g), float(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

R, _ = orthogonal_procrustes(x, rec.detach().numpy())
rec_proposed = rec.detach().numpy() @ R.T
loss_proposed = float(dG(torch.FloatTensor(x), rec))


net = GIN(m)
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for epoch in range(100):
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])
    data = Data(x=X_extended, edge_index=edge_index)
    rec = net(data)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])
    print(float(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

R, _ = orthogonal_procrustes(x, rec.detach().numpy())
rec_GIN = rec.detach().numpy() @ R.T
loss_GIN = float(dG(torch.FloatTensor(x), rec))

net = GAT(m)
optimizer = optim.Adam(net.parameters(), lr=0.001)
net.train()
for epoch in range(100):
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])
    data = Data(x=X_extended, edge_index=edge_index)
    rec = net(data)
    loss = dG(torch.FloatTensor(x)[train_ind], rec[train_ind])
    print(float(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

R, _ = orthogonal_procrustes(x, rec.detach().numpy())
rec_GAT = rec.detach().numpy() @ R.T
loss_GAT = float(dG(torch.FloatTensor(x), rec))

ind = torch.eye(n)[:, torch.randperm(n)[:m]]
X_extended = torch.hstack([X, ind])
X_embedded = TSNE(n_components=2, random_state=0, init='pca').fit_transform(X_extended.numpy())
loss_tSNE = float(dG(torch.FloatTensor(x), X_embedded))


c = x[:, 0].argsort().argsort()
fig = plt.figure(figsize=(14, 4))
ax = fig.add_subplot(2, 3, 1)
ax.scatter(x[:, 0], x[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('#eeeeee')
txt = ax.text(0.05, 0.05, 'Ground Truth', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='#eeeeee')])

visible = plt.imread('./imgs/visible.png')
visible_ax = fig.add_axes([0.24, 0.77, 0.1, 0.1], anchor='NE', zorder=1)
visible_ax.imshow(visible)
visible_ax.axis('off')

G = nx.DiGraph()
G.add_edges_from([(fr[i], to[i]) for i in range(len(fr))])
ax = fig.add_subplot(2, 3, 2)
pos = nx.spring_layout(G, k=0.18, seed=0)
nx.draw_networkx(G, ax=ax, pos=pos, node_size=0.5, node_color='#005aff', labels={i: '' for i in range(n)}, edge_color='#84919e', width=0.0005, arrowsize=0.1)
txt = ax.text(0.05, 0.05, 'Input Graph', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
ax.set_rasterization_zorder(3)

ax = fig.add_subplot(2, 3, 3)
ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
txt = ax.text(0.05, 0.05, 'tSNE(X) $d_G = {:.2f}$'.format(loss_tSNE), color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

ax = fig.add_subplot(2, 3, 4)
ax.scatter(rec_proposed[:, 0], rec_proposed[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
txt = ax.text(0.05, 0.05, 'Proposed $d_G = \\mathbf{' + f'{loss_proposed:.3f}' + '}$', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

ax = fig.add_subplot(2, 3, 5)
ax.scatter(rec_GIN[:, 0], rec_GIN[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
txt = ax.text(0.05, 0.05, 'GIN $d_G = {:.2f}$'.format(loss_GIN), color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

ax = fig.add_subplot(2, 3, 6)
ax.scatter(rec_GAT[:, 0], rec_GAT[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
txt = ax.text(0.05, 0.05, 'GAT $d_G = {:.2f}$'.format(loss_GAT), color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

fig.subplots_adjust()

if not os.path.exists('imgs'):
    os.mkdir('imgs')

fig.savefig('imgs/semi_moon.png', bbox_inches='tight', dpi=300)
fig.savefig('imgs/semi_moon.pdf', bbox_inches='tight', dpi=300)
fig.savefig('imgs/semi_moon.svg', bbox_inches='tight', dpi=300)
