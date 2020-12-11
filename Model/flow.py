from Model import models
from sklearn.metrics import precision_score
import torch
import pickle
import dgl
import random
import logging
import os


def collate_long(samples):
    train_graphs, train_feats, labels, scores = map(list, zip(*samples))
    batch_graph = dgl.batch(train_graphs)
    batch_feats = torch.tensor(train_feats, dtype=torch.float32)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    batch_scores = torch.tensor(scores, dtype=torch.float32)
    return batch_graph, batch_feats, batch_labels, batch_scores


def collate_float(samples):
    train_graphs, train_feats, labels, scores = map(list, zip(*samples))
    batch_graph = dgl.batch(train_graphs)
    batch_feats = torch.tensor(train_feats, dtype=torch.float32)
    batch_labels = torch.tensor(labels, dtype=torch.float32)
    batch_scores = torch.tensor(scores, dtype=torch.float32)
    return batch_graph, batch_feats, batch_labels, batch_scores


def lossfunc(true_labels, predicted_labels, true_scores, predicted_scores):
    Crossloss = torch.nn.CrossEntropyLoss()
    Mseloss = torch.nn.MSELoss()
    mask = torch.eq(true_labels, 1)
    predicted_scores_selected = torch.masked_select(
        torch.squeeze(predicted_scores), mask)
    true_scores_selected = torch.masked_select(true_scores, mask)
    loss = Crossloss(
        predicted_labels, true_labels)+Mseloss(predicted_scores_selected, true_scores_selected)*3
    return loss


def train_classification(model, train_datas, val_datas, batchsize, path, epoch):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(filename=path+"/log",
                        level=logging.DEBUG, filemode='w')

    # 数据已经经过了sigmoid，所以只需要log likelihood loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for i in range(1, epoch+1):
        train_epoch_loss = []
        train_data_loader = torch.utils.data.DataLoader(
            train_datas, batch_size=batchsize, shuffle=True, collate_fn=collate_long)
        for train_graphs, train_feats, train_labels, train_scores in train_data_loader:
            optimizer.zero_grad()
            predicted_labels, predicted_scores = model(
                train_graphs, train_feats)
            train_loss = lossfunc(
                train_labels, predicted_labels, train_scores, predicted_scores)
            train_temp = torch.nn.Softmax()(predicted_labels)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss.append(train_loss.detach().item())  # 每一个批次的损失

        # validata
        val_graphs, val_feats, val_labels, val_scores = collate_long(
            val_datas)  # 产生一个batch
        val_predictions, val_pred_score = model(val_graphs, val_feats)
        val_epoch_loss = lossfunc(
            val_labels, val_predictions, val_scores, val_pred_score)

        val_maxindexs = torch.argmax(
            torch.nn.Softmax()(val_predictions), -1).detach()
        val_metrix = precision_score(
            val_labels, val_maxindexs, average="macro")
        msg = 'epoch {} train_loss:{}'.format(i, sum(train_epoch_loss)/len(
            train_epoch_loss)) + 'val_loss:{}'.format(val_epoch_loss) + 'val metrix:{}'.format(val_metrix)
        logging.info(msg)
        print(msg)

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


def select_classification(model, datas):
    select_data_loader = torch.utils.data.DataLoader(
        datas, batch_size=128, shuffle=True, collate_fn=collate_long)
    res = []
    for select_graphs, select_feats, select_labels, select_scores in select_data_loader:
        predictions, scores = model(select_graphs, select_feats)
        predictions = torch.nn.Softmax()(predictions)
        for index, item in enumerate(predictions):
            if predictions[index][0] == max(predictions[index]) or scores[index] >= 0.2:
                res.append(True)
            # if item[0] >= thred:
            #     res.append(True)
            else:
                res.append(False)
    return res


if __name__ == "__main__":
    pass
