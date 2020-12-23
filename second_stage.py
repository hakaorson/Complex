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
    args.modelname = args.modelpath
    model = models.get_model(args)
    model.load_state_dict(torch.load(
        "Model/saved_models/{}/{}.pt".format(args.modelpath, args.modelepoch)))
    res, truescores, predictscores = flow.expand_selection(model, datas)
    # 需要分析一下
    expand_datas_selected = []
    truescores_selected = []
    predictscores_selected = []
    for index, val in enumerate(res):
        if val:
            expand_datas_selected.append(expand_datas[index])
            truescores_selected.append(truescores[index])
            predictscores_selected.append(predictscores[index])
    predict_true_count = 0
    for score in truescores_selected:
        if score >= 0.25:
            predict_true_count += 1
    print(predict_true_count/len(predictscores_selected))

    expand_f1computor = metrix.ClusterQualityF1(
        bench_datas, expand_datas_selected, metrix.NAAffinity, 0.25)
    expand_score = expand_f1computor.score()
    print(expand_score)

    normal_f1computor = metrix.ClusterQualityF1(
        bench_datas, normal_datas, metrix.NAAffinity, 0.25)
    normal_score = normal_f1computor.score()
    print(normal_score)


if __name__ == "__main__":
    args = first_stage.arg_parse()
    print(args)
    CheckModel(args)
