# Generate Figure 2
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
# %%


def bfs(s, t, x, K):
    n = len(x)
    dist = [1e9 for i in range(n)]
    dist[s] = 0
    prev = [None for i in range(n)]
    que = [s]
    while len(que) >= 1:
        cur = que.pop(0)
        neigh = np.argsort(((x - x[cur]) ** 2).sum(axis=1))[1:K + 1]
        for r in neigh:
            if dist[r] > 1e8:
                prev[r] = cur
                dist[r] = dist[cur] + 1
                que.append(r)
    cur = t
    res = []
    while cur is not None:
        res.append(cur)
        cur = prev[cur]
    return dist[t], res


# %%
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['pdf.fonttype'] = 42
# %%
# Generate data
np.random.seed(0)
p = np.array([
    [-2.0, -1.0],
    [1.0, -3.0],
    [8.0, 1.0]
])  # Landmark nodes (A, B, C)
t = np.pi * np.random.rand(8000, 1)
x = 6 * np.cos(t)
y = 6 * np.sin(t)
z = np.hstack([x, y])
a = np.random.randn(1000, 2) + z[:1000]
b = np.random.randn(2000, 2) * 0.4 + z[:2000]
c = np.random.randn(1000, 2) + np.array([6, 0]) + z[:1000] * np.array([1, -1])
d = np.random.randn(2000, 2) * 0.4 + np.array([6, 0]) + z[:2000] * np.array([1, -1])
e = np.random.randn(1000, 2) * 0.6 + np.array([1, -3])
x = np.concatenate([p, a, b, c, d, e])
# %%
plt.scatter(x[:, 0], x[:, 1], s=5, marker='o', c='#005aff')
# %%
d1, li1 = bfs(0, 1, x, 10)
d2, li2 = bfs(0, 2, x, 10)
# %%
d1, d2
# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x[:, 0], x[:, 1], s=5, marker='o', c='#005aff')
ax.scatter(x[:3, 0], x[:3, 1], s=50, c='#ff4b00')
ax.plot(x[li1, 0], x[li1, 1], c='#ff4b00')
ax.plot(x[li2, 0], x[li2, 1], c='#ff4b00')
txt = ax.text(1, -4.5, '{} hops'.format(d1), color='#ff4b00', fontsize=14, weight='bold')
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
txt = ax.text(8.5, 0, '{} hops'.format(d2), color='#ff4b00', fontsize=14, weight='bold')
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
for i in range(3):
    txt = ax.text(x[i, 0] - 0.8, x[i, 1] + 0.5, ['A', 'B', 'C'][i], color='#000000', fontsize=16, weight='bold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
ax.set_xticks([])
ax.set_yticks([])
if not os.path.exists('imgs'):
    os.mkdir('imgs')
fig.savefig('imgs/shortest_path.png', bbox_inches='tight')
fig.savefig('imgs/shortest_path.pdf', bbox_inches='tight')
