from Model import models
from sklearn.metrics import precision_score
import torch
import pickle
import dgl
import random
import logging
import numpy as np
import os
from sklearn import svm
from sklearn import metrics
import joblib


def collate_long(samples):
    train_graphs, feats, labels, score_prediction = map(list, zip(*samples))
    batch_graph = dgl.batch(train_graphs)
    batch_feats = torch.stack(feats, 0)
    batch_labels = torch.stack(labels, 0)
    batch_scores = torch.stack(score_prediction, 0)
    return batch_graph, batch_feats, batch_labels, batch_scores


def lossfunc(true_labels, predicted_labels, true_scores, score_prediction):
    Crossloss = torch.nn.CrossEntropyLoss()
    Mseloss = torch.nn.MSELoss()
    mask = torch.gt(true_labels, -1)  # 计算所有的loss
    score_prediction_selected = torch.masked_select(
        torch.squeeze(score_prediction), mask)
    true_scores_selected = torch.masked_select(true_scores, mask)
    return Crossloss(predicted_labels, true_labels), Mseloss(score_prediction_selected, true_scores_selected)


def train_classification(model, train_datas, val_datas, batchsize, path, epoch, cudaindex, lr):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(filename=path+"/log",
                        level=logging.DEBUG, filemode='w')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0.1)  # 这里设置了l2正则化系数
    model.train()
    for i in range(1, epoch+1):
        train_epoch_loss = []
        train_data_loader = torch.utils.data.DataLoader(
            train_datas, batch_size=batchsize, shuffle=True, collate_fn=collate_long)
        for train_graphs, train_feats, train_labels, train_scores in train_data_loader:
            optimizer.zero_grad()
            predicted_labels, score_prediction = model(
                train_graphs, train_feats)
            traincrossloss, trainmseloss = lossfunc(
                train_labels, predicted_labels, train_scores, score_prediction)
            trainlosssum = traincrossloss + trainmseloss*10  # 增强回归的损失
            trainlosssum.backward()
            optimizer.step()
            train_epoch_loss.append(
                [traincrossloss.detach().item(), trainmseloss.detach().item()])  # 每一个批次的损失

        # validata
        val_graphs, val_feats, val_labels, val_scores = collate_long(
            val_datas)  # 产生一个batch
        val_predictions, val_pred_score = model(val_graphs, val_feats)
        valcrossloss, valmseloss = lossfunc(
            val_labels, val_predictions, val_scores, val_pred_score)

        val_maxindexs = torch.argmax(val_predictions, -1).detach()
        train_maxindexs = torch.argmax(predicted_labels, -1).detach()
        val_metrix = precision_score(
            val_labels, val_maxindexs, average="macro")
        trainlosses = np.array(train_epoch_loss)
        msg = 'epoch {} train_loss:{}'.format(i, np.mean(
            trainlosses, 0)) + 'val_loss:[{} {}]'.format(valcrossloss, valmseloss) + 'val metrix macro:{}'.format(val_metrix)
        logging.info(msg)
        print(msg)
        print("last train batch fusion")
        print(metrics.confusion_matrix(train_labels, train_maxindexs))
        print("val fusion")
        print(metrics.confusion_matrix(val_labels, val_maxindexs))

        # 存储模型
        if i != 0 and i % 20 == 0:
            torch.save(model.state_dict(), path+'/{}.pt'.format(i))


def train_svmclassify(model, train_datas, val_datas, batchsize, path, epoch):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(filename=path+"/log",
                        level=logging.DEBUG, filemode='w')
    clf = svm.SVC(C=0.3, kernel='linear', gamma=20,
                  decision_function_shape='ovr')
    train_graphs, train_feats, train_labels, train_scores = collate_long(
        train_datas)
    deepwalkfeats = model(train_graphs, train_feats).numpy()
    train_labels = np.array(train_labels)
    clf.fit(deepwalkfeats, train_labels)
    train_predicts = clf.predict(deepwalkfeats)
    train_f1 = metrics.f1_score(
        train_predicts, train_labels, average='macro')

    val_graphs, val_feats, val_labels, val_scores = collate_long(
        val_datas)  # 产生一个batch
    deepwalkfeats_val = model(val_graphs, val_feats).numpy()
    val_labels = np.array(val_labels)
    val_predicts = clf.predict(deepwalkfeats_val)
    val_f1 = metrics.f1_score(val_predicts, val_labels, average='macro')
    # clf.fit(deepwalkfeats_val, val_labels)
    val_predicts_second = clf.predict(deepwalkfeats_val)
    val_f1_second = metrics.f1_score(
        val_predicts_second, val_labels, average='macro')
    msg = 'train_metrix:{} '.format(
        train_f1) + 'val metrix:{}'.format(val_f1)
    logging.info(msg)
    joblib.dump(clf, path+'/sklearn_model')
    print(msg)


def expand_selection(model, datas):
    expand_data_loader = torch.utils.data.DataLoader(
        datas, batch_size=128, shuffle=False, collate_fn=collate_long)  # 注意这里一定不能shuffle，否在数据就会不一致
    res = []
    truescores = []
    predictscores = []
    for index, (expand_graphs, expand_feats, expand_labels, expand_scores) in enumerate(expand_data_loader):
        class_prediction, score_prediction = model(expand_graphs, expand_feats)
        class_prediction = torch.nn.Softmax()(class_prediction)
        score_prediction = torch.squeeze(score_prediction)
        maxindexs = torch.argmax(class_prediction, -1).detach()
        print("fusion {}\n".format(index),
              metrics.confusion_matrix(expand_labels, maxindexs))
        for index, predict_label in enumerate(maxindexs):
            predict_score = score_prediction[index]
            truescores.append(expand_scores[index])
            predictscores.append(predict_score)
            if predict_label == 0 or predict_label == 2 or predict_score >= 0.25:
                res.append(True)
            else:
                res.append(False)
    return res, truescores, predictscores


if __name__ == "__main__":
    pass
