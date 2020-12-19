from dgl.data.graph_serialize import load_graphs
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import citation_graph as citegrh
import networkx as nx
import pandas as pd
import dgl
import numpy as np
import time

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(1433, 160)
        self.layer2 = GCNLayer(160, 16)
        self.layer3 = GCNLayer(16, 6)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = F.relu(self.layer2(g, x))
        x = self.layer3(g, x)
        return x


def load_graph():
    # pandas reads csv
    edges_data = pd.read_csv('data/knowledge_aquisition_reference.csv')
    # networkx reads pandas
    g_nx: nx.DiGraph = nx.from_pandas_edgelist(edges_data,
                                               'paper_id',
                                               'reference_id',
                                               create_using=nx.DiGraph())
    # dgl read networkx
    # ATTENTION!!!: nodes in dgl graph is ordered by paperid
    return dgl.from_networkx(g_nx)


def generate_features(g):
    return th.FloatTensor(np.zeros((g.number_of_nodes(), 1433)))


def load_labels():
    label_pd = pd.read_csv('data/rank_id.csv')
    return th.LongTensor(label_pd['reference_count'])


def generate_mask(g, labels):
    train_m = [
        (labels[i] != 0 and i % 10 <= 1) for i in range(g.number_of_nodes())
    ]
    test_m = [
        (labels[i] != 0 and i % 10 > 1) for i in range(g.number_of_nodes())
    ]
    train_mask = th.BoolTensor(train_m)
    test_mask = th.BoolTensor(test_m)
    return train_mask, test_mask


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def prepare_train():
    g = load_graph()
    features = generate_features(g)
    labels = load_labels()
    train_mask, test_mask = generate_mask(g, labels)
    return g, features, labels, train_mask, test_mask


def train(net, g, features, labels, train_mask, test_mask, lr=0.04, round=100):
    optimizer = th.optim.Adam(net.parameters(), lr=lr)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    dur = []
    for epoch in range(round):
        t0 = time.time()

        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        dur.append(time.time() - t0)

        acc = evaluate(net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".
              format(epoch, loss.item(), acc, np.mean(dur)))


if __name__ == '__main__':
    g, features, labels, train_mask, test_mask = prepare_train()
    net = Net()
    train(net, g, features, labels, train_mask, test_mask, lr=0.04, round=100)
