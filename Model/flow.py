from Model import graph_classify
from sklearn.metrics import precision_score
import torch
import pickle
import dgl
import random
import os


def collate(samples):
    train_graphs, train_feats, labels = map(list, zip(*samples))
    batch_graph = dgl.batch(train_graphs)
    batch_feats = torch.tensor(train_feats, dtype=torch.float32)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return batch_graph, batch_feats, batch_labels


def train_classification(model, train_datas, val_datas, batchsize, path, epoch):
    cross_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for i in range(1, epoch+1):
        train_epoch_loss = []
        train_data_loader = torch.utils.data.DataLoader(
            train_datas, batch_size=batchsize, shuffle=True, collate_fn=collate)
        for train_graphs, train_feats, train_labels in train_data_loader:
            train_prediction = model(train_graphs, train_feats)
            train_loss = cross_loss(train_prediction, train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_epoch_loss.append(train_loss.detach().item())  # 每一个批次的损失

        # validata
        val_graphs, val_feats, val_labels = collate(val_datas)  # 产生一个batch
        val_predictions = model(val_graphs, val_feats)
        val_epoch_loss = cross_loss(
            val_predictions, val_labels).detach().item()

        val_maxindexs = torch.argmax(torch.nn.Softmax()(
            val_predictions), -1).detach()
        val_metrix = precision_score(
            val_labels, val_maxindexs, average="micro")
        print('epoch {} train_loss:'.format(i), sum(train_epoch_loss)/len(train_epoch_loss),
              'val train_loss:', val_epoch_loss,
              'val metrix:', val_metrix)

        # 存储模型
        if i != 0 and i % 2 == 0:
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), path+'/{}.pt'.format(i))


def select(model, datas, thred=0.3):
    select_graphs, select_feats, _ = collate(datas)
    predictions = torch.nn.Softmax()(model(select_graphs, select_feats))
    res = []
    for item in predictions:
        if item[0] >= 0.3:
            res.append(True)
        else:
            res.append(False)
    return res


if __name__ == "__main__":
    pass
