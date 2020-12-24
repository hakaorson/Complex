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
from Check import metrix


def arg_parse():
    '''
    argument parser
    '''
    parser = argparse.ArgumentParser(description='Train model arguments')
    parser.add_argument('--recompute', type=int, default=0)
    parser.add_argument('--refer', type=str)
    parser.add_argument('--bench', type=str)
    parser.add_argument('--graph', type=str)
    parser.add_argument('--rebalance', type=int, default=1)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--info', type=str, default="")
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--modelname', type=str)

    parser.add_argument('--nodefeatsize', type=int, default=82)
    parser.add_argument('--edgefeatsize', type=int, default=19)
    parser.add_argument('--graphfeatsize', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=4)
    # 下面是加载模型参数
    parser.add_argument('--modelpath', type=str)
    parser.add_argument('--modelepoch', type=int)
    parser.add_argument('--sklearnmodel', type=int, default=0)
    return parser.parse_args()
    # return parser.parse_args("--bench Other --graph Biogrid --refer dpclus --modelpath Fusion_GCN_with_Topologi_Other_Biogrid_coach_U_final_12_24_11_16 --modelepoch 30 --modelname Fusion_GCN_with_Topologi".split(' '))
    # return parser.parse_args("--refer coach --bench Other --graph DIP --info final --modelname Only_struct_GCN --recompute 1".split(' '))


'''
python3 first_stage.py --refer coach --bench Other --graph Biogrid --info final --modelname Only_struct_GCN --recompute 0
python3 first_stage.py --refer coach --bench Other --graph Biogrid --info final --modelname Only_Node_feat_GCN --gcn_layers 1 --recompute 0
python3 first_stage.py --refer coach --bench Other --graph Biogrid --info final --modelname Only_Edge_feat_GCN --recompute 0
python3 first_stage.py --refer coach --bench Other --graph Biogrid --info final --modelname Fusion_GCN --recompute 0
python3 first_stage.py --refer coach --bench Other --graph Biogrid --info final --modelname Only_Topologi --recompute 0
python3 first_stage.py --refer coach --bench Other --graph Biogrid --info final --modelname Fusion_GCN_with_Topologi --recompute 0
python3 first_stage.py --refer coach --bench Other --graph Biogrid --info final --modelname GCN_with_Topologi_with_topk --recompute 0
python3 first_stage.py --refer coach --bench Other --graph Biogrid --info final --modelname GCN_with_Topologi_with_max --recompute 0

python3 first_stage.py --refer coach --bench Other --graph DIP --info final --modelname Only_struct_GCN --recompute 0
python3 first_stage.py --refer coach --bench Other --graph DIP --info final --modelname Only_Node_feat_GCN --gcn_layers 1 --recompute 0
python3 first_stage.py --refer coach --bench Other --graph DIP --info final --modelname Only_Edge_feat_GCN --recompute 0
python3 first_stage.py --refer coach --bench Other --graph DIP --info final --modelname Fusion_GCN --recompute 0
python3 first_stage.py --refer coach --bench Other --graph DIP --info final --modelname Only_Topologi --recompute 0
python3 first_stage.py --refer coach --bench Other --graph DIP --info final --modelname Fusion_GCN_with_Topologi --recompute 0
python3 first_stage.py --refer coach --bench Other --graph DIP --info final --modelname GCN_with_Topologi_with_topk --recompute 0
python3 first_stage.py --refer coach --bench Other --graph DIP --info final --modelname GCN_with_Topologi_with_max --recompute 0


python3 second_stage.py --bench Other --graph Biogrid --refer clique_percolation --recompute 0
python3 second_stage.py --bench Other --graph Biogrid --refer ipca --recompute 0
python3 second_stage.py --bench Other --graph Biogrid --refer mcode --recompute 0
python3 second_stage.py --bench Other --graph Biogrid --refer graph_entropy --recompute 0
python3 second_stage.py --bench Other --graph Biogrid --refer dpclus --recompute 0


python3 second_stage.py --bench Other --graph DIP --refer clique_percolation --recompute 0
python3 second_stage.py --bench Other --graph DIP --refer ipca --recompute 0
python3 second_stage.py --bench Other --graph DIP --refer mcode --recompute 0
python3 second_stage.py --bench Other --graph DIP --refer graph_entropy --recompute 0
python3 second_stage.py --bench Other --graph DIP --refer dpclus --recompute 0
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


def to_tensor(datas, cudaindex):
    res = []
    for graph, feat, label, score in datas:
        feat = torch.tensor(feat, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        score = torch.tensor(score, dtype=torch.float32)
        if cudaindex != -1:
            graph = graph.to(torch.device("cuda:{}".format(cudaindex)))
            feat = feat.cuda("cuda:{}".format(cudaindex))
            label = label.cuda("cuda:{}".format(cudaindex))
            score = score.cuda("cuda:{}".format(cudaindex))
        res.append([graph, feat, label, score])
    return res


def classification_process(args):
    datasets_list_by_class, datasets_name = data.trainmodel_datasets(
        basedir="Data/", recompute=args.recompute, refername=args.refer, benchname=args.bench, graphname=args.graph)
    print([len(item) for item in datasets_list_by_class])
    train_datasets, val_datasets = split_datasets(
        datasets_list_by_class, args.split)
    if args.rebalance:
        train_datasets = rebalance_datasets(train_datasets)
    train_datasets = [[item.graph, item.feat, item.label, item.score]
                      for item in reduce(lambda a, b: a+b, train_datasets)]
    val_datasets = [[item.graph, item.feat, item.label, item.score]
                    for item in reduce(lambda a, b: a+b, val_datasets)]

    model = models.get_model(args)

    if args.cuda >= 0:
        model = model.cuda("cuda:{}".format(args.cuda))
    train_datasets = to_tensor(train_datasets, args.cuda)
    val_datasets = to_tensor(val_datasets, args.cuda)

    model_path = "Model/saved_models/{}_{}_{}_{}".format(args.modelname, datasets_name, args.info,
                                                         time.strftime('{}_{}_{}_{}'.format(time.localtime().tm_mon, time.localtime().tm_mday, ((time.localtime().tm_hour)+8) % 24, time.localtime().tm_min)))
    flow.train_classification(model, train_datasets, val_datasets,
                              args.batchsize, model_path, args.epoch, args.cuda, args.lr)


if __name__ == "__main__":
    args = arg_parse()
    print(args)
    classification_process(args)
