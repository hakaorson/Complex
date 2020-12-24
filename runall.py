import os
import subprocess
import time
from first_stage import arg_parse
targetgraph = ["DIP", "Biogrid"]

referMethods = ["dpclus", "clique_percolation",
                "graph_entropy", "ipca", "mcode"]


def run_all_train():
    '''
    python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Only_struct_GCN --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Only_Node_feat_GCN --gcn_layers 1 --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Only_Edge_feat_GCN --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Fusion_GCN --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Only_Topologi --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Fusion_GCN_with_Topologi --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname GCN_with_Topologi_with_topk --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname GCN_with_Topologi_with_max --recompute 0 --epoch 80

    python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Only_struct_GCN --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Only_Node_feat_GCN --gcn_layers 1 --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Only_Edge_feat_GCN --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Fusion_GCN --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Only_Topologi --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Fusion_GCN_with_Topologi --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname GCN_with_Topologi_with_topk --recompute 0 --epoch 80
    python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname GCN_with_Topologi_with_max --recompute 0 --epoch 80
    '''
    all_cmds = [
        "python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Only_struct_GCN --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Only_Node_feat_GCN --gcn_layers 1 --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Only_Edge_feat_GCN --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Fusion_GCN --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Only_Topologi --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname Fusion_GCN_with_Topologi --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname GCN_with_Topologi_with_topk --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph Biogrid --info 80epoch --modelname GCN_with_Topologi_with_max --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Only_struct_GCN --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Only_Node_feat_GCN --gcn_layers 1 --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Only_Edge_feat_GCN --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Fusion_GCN --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Only_Topologi --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname Fusion_GCN_with_Topologi --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname GCN_with_Topologi_with_topk --recompute 0 --epoch 80",
        "python3 first_stage.py --refer coach --bench Other --graph DIP --info 80epoch --modelname GCN_with_Topologi_with_max --recompute 0 --epoch 80"
    ]
    for cmd in all_cmds:
        p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE)
        while p.poll() is None:
            line = p.stdout.readline().decode("utf8").strip()
            if len(line.strip()):  # 空行需要删除
                print(line)
        time.sleep(5)


def run_all_test():
    modelbasepath = "Model/saved_models"
    for nowgraphname in targetgraph:
        for modelpath in os.listdir(modelbasepath):
            infolist = list(modelpath.split('_'))
            modelname = '_'.join(infolist[:-9])
            benchname = infolist[-9]
            graphname = infolist[-8]
            epoch = 80
            if nowgraphname != graphname or "final" in infolist:  # 模型的图需要和现在测试的图一致
                continue
            for ref in referMethods:
                allres = []
                cmd = "python3 second_stage.py --bench {} --graph {} --refer {} --modelpath {} --modelepoch {} --modelname {}".format(
                    benchname, graphname, ref, modelpath, epoch, modelname)
                print(cmd)
                p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE)
                while p.poll() is None:
                    line = p.stdout.readline().decode("utf8").strip()
                    if len(line.strip()):  # 空行需要删除
                        # print(line)
                        allres.append(line)
                time.sleep(5)


if __name__ == "__main__":
    # run_all_train()
    run_all_test()
