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
from sklearn.externals import joblib


def arg_parse():
    '''
    argument parser
    '''
    parser = argparse.ArgumentParser(description='Train model arguments')
    parser.add_argument('--recompute', type=int, default=0)
    parser.add_argument('--refer', type=str, default="coach")
    parser.add_argument('--bench', type=str, default="CYC2008")
    parser.add_argument('--graph', type=str, default="DIP")
    parser.add_argument('--refer_rate', type=int, default=1)
    parser.add_argument('--random_rate', type=int, default=1)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--modelpath', type=str)
    parser.add_argument('--modelepoch', type=int)
    parser.add_argument('--modeltype', type=str, default="sklearn")
    return parser.parse_args()
    # return parser.parse_args("--refer coach --bench CYC2008 --graph DIP --recompute 0 --modelpath onlydeepwalk_CYC2008_Biogrid_coach_U_12_11_13_28".split(" "))


'''
python second_stage.py --modeltype pytorch --refer coach --bench CYC2008 --graph Biogrid --recompute 0 --modelpath gcnwithtopo_CYC2008_Biogrid_coach_U_12_11_11_44 --modelepoch 6
python second_stage.py --refer coach --bench CYC2008 --graph Biogrid --recompute 0 --modelpath onlydeepwalk_CYC2008_Biogrid_coach_U_12_11_13_28


'''


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


if __name__ == "__main__":
    args = arg_parse()
    print(args)
    normal_datas, expand_datas, datas = data.selectcomplex_datasets(
        basedir="Data/", recompute=args.recompute, refername=args.refer, graphname=args.graph)
    datas = [[item.graph, item.feat, item.label, item.score]
             for item in datas]
    model = models.OnlyDeepwalk(
        nodefeatsize=66,
        edgefeatsize=19,
        graphfeatsize=10,
        hidden_size=128,
        gcn_layers=2,
        output_size=3,
        activate=None
    )
    if args.modeltype == "sklearn":
        skmodel = joblib.load(
            "Model/saved_models/{}/sklearn_model".format(args.modelpath))
        model = warpedmodel(skmodel, model)
    else:
        model.load_state_dict(torch.load(
            "Model/saved_models/{}/{}.pt".format(args.modelpath, args.modelepoch)))
    res = flow.select_classification(model, datas)
    # 需要分析一下
    expand_datas_selected = []
    for index, val in enumerate(res):
        if val:
            expand_datas_selected.append(expand_datas[index])

    bench_datas = data.read_complexes("Data/bench/CYC2008/complexes")

    normal_f1computor = metrix.ClusterQualityF1(
        bench_datas, normal_datas, metrix.NAAffinity, 0.25)
    normal_score = normal_f1computor.score()

    expand_f1computor = metrix.ClusterQualityF1(
        bench_datas, expand_datas_selected, metrix.NAAffinity, 0.25)
    expand_score = expand_f1computor.score()
    print(normal_score, expand_score)
