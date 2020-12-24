from Data import data
from Model import models
from Model import flow
import random
import numpy as np
import torch
import time
import pickle
import os
import subprocess
from Check import metrix
import argparse
import joblib
import first_stage
from sklearn import metrics


class warpedmodel():
    def __init__(self, skmodel, pytorchmodel):
        self.skmodel = skmodel
        self.pytorchmodel = pytorchmodel

    def __call__(self, graphs, feats):
        pyt = self.pytorchmodel(graphs, feats)
        svc_res = self.skmodel.predict(pyt)
        res = [[0, 0, 0] for i in range(len(svc_res))]
        for i in range(len(svc_res)):
            res[i][svc_res[i]] = 1
        return np.array(res), [0 for i in range(len(svc_res))]


def CheckModel(args):
    normal_datas, expand_datas, datas = data.selectcomplex_datasets(
        basedir="Data/", recompute=args.recompute, refername=args.refer, graphname=args.graph, benchname=args.bench)
    bench_datas = data.read_complexes(
        "Data/bench/{}/complexes".format(args.bench))

    datas = [[item.graph, item.feat, item.label, item.score]
             for item in datas]
    datas = first_stage.to_tensor(datas, args.cuda)
    model = models.get_model(args)
    model.load_state_dict(torch.load(
        "Model/saved_models/{}/{}.pt".format(args.modelpath, args.modelepoch)), False)
    truescores, predictscores, truelabels, predictlabels = flow.expand_selection(
        model, datas, args.batchsize)

    allres = []
    allres.append(str(metrics.confusion_matrix(truelabels, predictlabels)))
    # 需要分析一下
    expand_datas_selected = []
    truescores_selected = []
    predictscores_selected = []
    for index in range(len(truescores)):
        if predictlabels == 0 or predictscores[index] >= 0.20:
            expand_datas_selected.append(expand_datas[index])
            truescores_selected.append(truescores[index])
            predictscores_selected.append(predictscores[index])
    allres.append("before select total num:{}".format(len(normal_datas)))
    normal_f1score = metrix.ClusterQualityF1_MMR(
        bench_datas, normal_datas, metrix.NAAffinity, 0.25).score()
    allres.append("recall precision f1 mmr:{}".format(normal_f1score))
    normal_SNscore = metrix.ClusterQualitySN_PPV_Acc(
        bench_datas, normal_datas).score()
    allres.append("recall precision sn ppv acc:{}".format(normal_SNscore))

    allres.append("after select total num:{}".format(
        len(expand_datas_selected)))
    expand_f1score = metrix.ClusterQualityF1_MMR(
        bench_datas, expand_datas_selected, metrix.NAAffinity, 0.25).score()
    allres.append("recall precision f1 mmr:{}".format(expand_f1score))
    expand_SNscore = metrix.ClusterQualitySN_PPV_Acc(
        bench_datas, expand_datas_selected).score()
    allres.append("recall precision sn ppv acc:{}".format(expand_SNscore))

    result_path = "Data/datasets/predict_res/"+args.modelpath+"_"+args.refer
    print(allres)
    with open(result_path, 'w') as f:
        for line in allres:
            f.write(line+'\n')


if __name__ == "__main__":
    args = first_stage.arg_parse()
    print(args)
    CheckModel(args)
    # CheckModel(args)
