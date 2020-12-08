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


if __name__ == "__main__":
    random.seed(666)
    normal_datas, expand_datas, datas = data.selectcomplex_datasets(
        basedir="Data/", recompute=True, refername="coach", typ="classification")
    datas = [[item.graph, item.feat, item.label] for item in datas]
    model = models.GCN_with_Topologi(
        nodefeatsize=486,
        edgefeatsize=19,
        graphfeatsize=10,
        hidden_size=128,
        gcn_layers=2,
        output_size=3,
        activate=None
    )
    model.load_state_dict(torch.load(
        r"Model/saved_models/gcnwithtopo_CYC2008_DIP_coach_classification_U_12081355/8.pt"))
    res = flow.select_classification(model, datas, thred=0.1)
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
