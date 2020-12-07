from Data import data
from Model import graph_classify
from Model import flow
import random
import torch
import time
import pickle

if __name__ == "__main__":
    random.seed(666)
    tarindatas = data.trainmodel_datasets(basedir="Data/", recompute=False)
    datas = [[item.graph, item.feat, item.label] for item in tarindatas]
    random.shuffle(datas)
    size = len(datas)
    cut1, cut2 = int(0.7*size), int(0.85*size)
    traindatas, valdatas, testdatas = datas[:
                                            cut1], datas[cut1:cut2], datas[cut2:]

    nodefeatsize = 486
    edgefeatsize = 19
    graphfeatsize = 10
    batchsize = 16
    model = graph_classify.GCN_with_Topologi_classification(
        nodefeatsize=nodefeatsize,
        edgefeatsize=edgefeatsize,
        graphfeatsize=graphfeatsize,
        hidden_size=128,
        gcn_layers=2,
        class_num=3
    )
    model_path = "Model/saved_models/{}_{}".format(model.name,
                                                   time.strftime('%m_%d_%H_%M', time.localtime()))
    default_epoch = 50

    flow.train_classification(model, traindatas, valdatas,
                              batchsize, model_path, default_epoch)
    model.load_state_dict(torch.load(
        model_path+'/{}.pt'.format(default_epoch)))
    # model.load_state_dict(torch.load(
    #     "Model/saved_models_base_11_28_19_39/10.pt"))
    flow.test(model, testdatas)
