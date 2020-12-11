from Data import data
from Model import models
from Model import flow
import random
import torch
import time
import pickle
import os
import subprocess
from Check import metrix
import argparse


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
    return parser.parse_args()
    #return parser.parse_args("--modelpath gcnwithtopo_CYC2008_DIP_coach_U_12_11_3_0 --modelepoch 50".split(" "))


'''
python second_stage.py --refer coach --bench CYC2008 --graph Biogrid --recompute 0 --modelpath gcnwithtopo_CYC2008_Biogrid_coach_U_12_11_4_53 --modelepoch 50
'''

if __name__ == "__main__":
    args = arg_parse()
    print(args)
    normal_datas, expand_datas, datas = data.selectcomplex_datasets(
        basedir="Data/", recompute=args.recompute, refername=args.refer, graphname=args.graph)
    datas = [[item.graph, item.feat, item.label, item.score]
             for item in datas]
    model = models.GCN_with_Topologi(
        nodefeatsize=66,
        edgefeatsize=19,
        graphfeatsize=10,
        hidden_size=128,
        gcn_layers=2,
        output_size=3,
        activate=None
    )
    model.load_state_dict(torch.load(
        "Model/saved_models/{}/{}.pt".format(args.modelpath, args.modelepoch)))
    res = flow.select_classification(model, datas)
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
