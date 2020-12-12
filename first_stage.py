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
from functools import reduce


def arg_parse():
    '''
    argument parser
    '''
    parser = argparse.ArgumentParser(description='Train model arguments')
    parser.add_argument('--recompute', type=int, default=0)
    parser.add_argument('--refer', type=str, default="coach")
    parser.add_argument('--bench', type=str, default="CYC2008")
    parser.add_argument('--graph', type=str, default="Krogan")
    parser.add_argument('--rebalance', type=int, default=0)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=666)
    return parser.parse_args()
    # return parser.parse_args("--refer coach --bench CYC2008 --graph DIP --recompute 1".split(' '))


'''
python3 first_stage.py --refer coach --bench CYC2008 --graph Biogrid --recompute 1
'''


def split_datasets(datas, rate):
    train, val = [], []
    for data in datas:
        cut = int(rate*len(data))
        train.append(data[:cut])
        val.append(data[cut:])
    return train, val


def rebalance_datasets(datas):
    res = []
    target = len(datas[0])
    for data in datas:
        res.append([])
        nums_multi, nums_left = target//len(data), target % len(data)
        res[-1].extend(list(data*nums_multi))
        res[-1].extend(random.choices(data, k=nums_left))
    return res


def classification_process(args):
    datasets_list_by_class, datasets_name = data.trainmodel_datasets(
        basedir="Data/", recompute=args.recompute, refername=args.refer, benchname=args.bench, graphname=args.graph)
    train_datasets, val_datasets = split_datasets(
        datasets_list_by_class, args.split)
    if args.rebalance:
        train_datasets = rebalance_datasets(train_datasets)
    train_datasets = [[item.graph, item.feat, item.label, item.score]
                      for item in reduce(lambda a, b: a+b, train_datasets)]
    val_datasets = [[item.graph, item.feat, item.label, item.score]
                    for item in reduce(lambda a, b: a+b, val_datasets)]

    model = models.GCN_with_Topologi(
        nodefeatsize=82,
        edgefeatsize=19,
        graphfeatsize=10,
        hidden_size=128,
        gcn_layers=2,
        output_size=5,
        activate=None
    )
    model_path = "Model/saved_models/{}_{}_{}".format(model.name, datasets_name,
                                                      time.strftime('{}_{}_{}_{}'.format(time.localtime().tm_mon, time.localtime().tm_mday, ((time.localtime().tm_hour)) % 24, time.localtime().tm_min)))
    default_epoch = 5000
    batchsize = 32
    flow.train_classification(model, train_datasets, val_datasets,
                              batchsize, model_path, default_epoch)


if __name__ == "__main__":
    args = arg_parse()
    print(args)
    classification_process(args)
