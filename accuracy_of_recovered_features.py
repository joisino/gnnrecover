import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.transforms.largest_connected_components import LargestConnectedComponents
from scipy.sparse import csr_matrix
from util import stationary, reconstruct_full
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


np.random.seed(0)
torch.manual_seed(0)

for dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'CS', 'Physics', 'Computers', 'Photo']:
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name, transform=LargestConnectedComponents())
    if dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root='/tmp/' + dataset_name, name=dataset_name, transform=LargestConnectedComponents())
    if dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root='/tmp/' + dataset_name, name=dataset_name, transform=LargestConnectedComponents())

    if 'train_mask' in dataset[0]:
        train_mask = dataset[0]['train_mask']
        val_mask = dataset[0]['val_mask']
        test_mask = dataset[0]['test_mask']
    else:
        n_train_per_class = 20
        n_val_per_class = 30
        n_classes = dataset[0]['y'].max() + 1
        train_mask = []
        val_mask = []
        test_mask = []
        for c in range(n_classes):
            all_c = np.where(dataset[0]['y'].numpy() == c)[0]
            if len(all_c) <= n_train_per_class + n_val_per_class:
                continue
            train, rest = train_test_split(all_c, train_size=n_train_per_class, random_state=0)
            val, test = train_test_split(rest, train_size=n_val_per_class, random_state=0)
            train_mask += train.tolist()
            val_mask += val.tolist()
            test_mask += test.tolist()
        train_mask = torch.LongTensor(train_mask)
        val_mask = torch.LongTensor(val_mask)
        test_mask = torch.LongTensor(test_mask)

    edge_index = dataset[0]['edge_index']
    n = dataset[0]['x'].shape[0]
    m = 500
    dim = 8  # Recovered feature dimension
    deg = torch.bincount(dataset[0]['edge_index'].reshape(-1), minlength=n)
    fr = np.concatenate([edge_index[0].numpy(), edge_index[1].numpy()])
    to = np.concatenate([edge_index[1].numpy(), edge_index[0].numpy()])
    A = csr_matrix(([1 / deg[x] for x in fr], (fr, to)))
    X = torch.tensor([[deg[i], n] for i in range(n)], dtype=torch.float)
    ind = torch.eye(n)[:, torch.randperm(n)[:m]]
    X_extended = torch.hstack([X, ind])

    pr = stationary(A)
    pr = np.maximum(pr, 1e-9)
    rec_orig = reconstruct_full(dim, deg.numpy(), pr, n, m, fr, to)
    rec_orig = torch.FloatTensor(rec_orig)

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(rec_orig[train_mask].numpy(), dataset[0]['y'][train_mask].numpy())
    print(dataset_name, 'Ours', (clf.predict(rec_orig[test_mask].numpy()) == dataset[0]['y'][test_mask].numpy()).mean())

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_extended[train_mask].numpy(), dataset[0]['y'][train_mask].numpy())
    print(dataset_name, 'Baseline', (clf.predict(X_extended[test_mask].numpy()) == dataset[0]['y'][test_mask].numpy()).mean())
