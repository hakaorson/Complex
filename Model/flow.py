from Model import models
from sklearn.metrics import precision_score
import torch
import pickle
import dgl
import random
import logging
import os


def collate_long(samples):
    train_graphs, train_feats, labels = map(list, zip(*samples))
    batch_graph = dgl.batch(train_graphs)
    batch_feats = torch.tensor(train_feats, dtype=torch.float32)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return batch_graph, batch_feats, batch_labels


def collate_float(samples):
    train_graphs, train_feats, labels = map(list, zip(*samples))
    batch_graph = dgl.batch(train_graphs)
    batch_feats = torch.tensor(train_feats, dtype=torch.float32)
    batch_labels = torch.tensor(labels, dtype=torch.float32)
    return batch_graph, batch_feats, batch_labels


def train_classification(model, train_datas, val_datas, batchsize, path, epoch):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(filename=path+"/log", level=logging.DEBUG)

    loss = torch.nn.CrossEntropyLoss()  # 数据已经经过了sigmoid，所以只需要log likelihood loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for i in range(1, epoch+1):
        train_epoch_loss = []
        train_data_loader = torch.utils.data.DataLoader(
            train_datas, batch_size=batchsize, shuffle=True, collate_fn=collate_long)
        for train_graphs, train_feats, train_labels in train_data_loader:
            train_prediction = model(train_graphs, train_feats)
            train_loss = loss(train_prediction, train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_epoch_loss.append(train_loss.detach().item())  # 每一个批次的损失

        # validata
        val_graphs, val_feats, val_labels = collate_long(
            val_datas)  # 产生一个batch
        val_predictions = model(val_graphs, val_feats)
        val_epoch_loss = loss(
            val_predictions, val_labels).detach().item()

        val_maxindexs = torch.argmax(
            torch.nn.Softmax()(val_predictions), -1).detach()
        val_metrix = precision_score(
            val_labels, val_maxindexs, average="macro")
        logging.info('epoch {} train_loss:{}'.format(i, sum(train_epoch_loss)/len(train_epoch_loss)) +
                     'val_loss:{}'.format(val_epoch_loss) +
                     'val metrix:{}'.format(val_metrix))

        # 存储模型
        if i != 0 and i % 2 == 0:
            torch.save(model.state_dict(), path+'/{}.pt'.format(i))


def train_regression(model, train_datas, val_datas, batchsize, path, epoch):
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for i in range(1, epoch+1):
        train_epoch_loss = []
        train_data_loader = torch.utils.data.DataLoader(
            train_datas, batch_size=batchsize, shuffle=True, collate_fn=collate_float)
        for train_graphs, train_feats, train_labels in train_data_loader:
            train_prediction = model(train_graphs, train_feats)
            train_loss = loss(train_prediction, train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_epoch_loss.append(train_loss.detach().item())  # 每一个批次的损失

        # validata
        val_graphs, val_feats, val_labels = collate_float(
            val_datas)  # 产生一个batch
        val_predictions = model(val_graphs, val_feats)
        val_epoch_loss = loss(
            val_predictions, val_labels).detach().item()

        print('epoch {} train_loss:'.format(i), sum(train_epoch_loss)/len(train_epoch_loss),
              'val_loss:', val_epoch_loss)

        # 存储模型
        if i != 0 and i % 2 == 0:
            os.makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), path+'/{}.pt'.format(i))


def select_classification(model, datas, thred=0.3):
    select_data_loader = torch.utils.data.DataLoader(
        datas, batch_size=128, shuffle=True, collate_fn=collate_long)
    res = []
    for select_graphs, select_feats, _ in select_data_loader:
        predictions = torch.nn.Softmax()(model(select_graphs, select_feats))
        for item in predictions:
            if item[0] == max(item) or item[3] == max(item) or item[4] == max(item):
                res.append(True)
            # if item[0] >= thred:
            #     res.append(True)
            else:
                res.append(False)
    return res


if __name__ == "__main__":
    pass
