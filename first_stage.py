from Data import data
from Model import models
from Model import flow
import random
import torch
import time
import datetime
import pickle
import logging
import argparse


def arg_parse():
    '''
    argument parser
    '''
    parser = argparse.ArgumentParser(description='Train model arguments')
    parser.add_argument('--recompute', type=int, default=0)
    parser.add_argument('--refer', type=str, default="coach")
    parser.add_argument('--bench', type=str, default="CYC2008")
    parser.add_argument('--graph', type=str, default="Krogan")
    parser.add_argument('--refer_rate', type=int, default=2)
    parser.add_argument('--random_rate', type=int, default=2)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=666)
    return parser.parse_args()

    # return parser.parse_args("--refer dpclus --bench CYC2008 --graph Biogrid --recompute 1".split(' '))
'''
python first_stage.py --refer coach --bench CYC2008 --graph Biogrid --recompute 1
'''


def construct_datasets(bench_datasets, refer_datasets, random_datasets, refer_rate, random_rate):
    res = []
    res.extend(bench_datasets)
    nums_multi, nums_left = (len(bench_datasets)*refer_rate)//len(
        refer_datasets), (len(bench_datasets)*refer_rate) % len(refer_datasets)
    res.extend(list(refer_datasets*nums_multi))
    res.extend(random.choices(refer_datasets, k=nums_left))

    nums_multi, nums_left = (len(bench_datasets)*random_rate)//len(
        random_datasets), (len(bench_datasets)*random_rate) % len(random_datasets)
    res.extend(list(random_datasets*nums_multi))
    res.extend(random.choices(random_datasets, k=nums_left))
    return res


def classification_process(args):
    bench_datasets, refer_datasets, random_datasets, datasets_name = data.trainmodel_datasets(
        basedir="Data/", recompute=args.recompute, refername=args.refer, benchname=args.bench, graphname=args.graph)
    bench_split = int(args.split*len(bench_datasets))
    refer_split = int(args.split*len(refer_datasets))
    random_split = int(args.split*len(random_datasets))
    tarindatas = construct_datasets(
        bench_datasets[:bench_split], refer_datasets[:refer_split], random_datasets[:random_split], args.refer_rate, args.random_rate)
    traindatas = [[item.graph, item.feat, item.label, item.score]
                  for item in tarindatas]
    valdatas = [[item.graph, item.feat, item.label, item.score]
                for item in bench_datasets[:bench_split]+refer_datasets[:refer_split]+random_datasets[:random_split]]
    random.shuffle(traindatas)
    random.shuffle(valdatas)

    model = models.GCN_with_Topologi(
        nodefeatsize=66,
        edgefeatsize=19,
        graphfeatsize=10,
        hidden_size=128,
        gcn_layers=2,
        output_size=3,
        activate=None
    )
    model_path = "Model/saved_models/{}_{}_{}".format(model.name, datasets_name,
                                                      time.strftime('{}_{}_{}_{}'.format(time.localtime().tm_mon, time.localtime().tm_mday, ((time.localtime().tm_hour)) % 24, time.localtime().tm_min)))
    default_epoch = 100
    batchsize = 16
    flow.train_classification(model, traindatas, valdatas,
                              batchsize, model_path, default_epoch)


if __name__ == "__main__":
    args = arg_parse()
    print(args)
    classification_process(args)
