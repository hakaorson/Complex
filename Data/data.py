import os
import pickle
import queue
import random
import shutil
import subprocess
from multiprocessing import pool as mtp

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import tqdm
from sklearn import preprocessing


from Data.refer import refermethods


def read_feat_datas(data_path):
    ids, datas = [], []
    with open(data_path, 'r') as f:
        featsnames = [item.split('_')[0]
                      for item in next(f).strip().split('\t')[1:]]
        featsnames = {name: [featsnames.index(name), featsnames.index(
            name)+featsnames.count(name)] for name in set(featsnames)}
        for item in f:
            item_splited = item.strip().split('\t')
            ids.append(item_splited[0])
            datas.append(list(map(float, item_splited[1:])))
    return ids, datas, featsnames


# 数据预处理，一些归一化等等
def dataprocess(matrix):
    matrix = np.array(matrix)
    matrix = preprocessing.normalize(matrix, axis=0)
    return matrix


def subgraphs(complexes, graph):
    res = []
    subgraph_notcomplete_num = 0
    all_complex_nodes = set()
    all_graph_nodes = set(graph.nodes)
    for comp in complexes:
        all_complex_nodes = all_complex_nodes | comp
        subgraph = nx.subgraph(graph, comp)
        subgraph_bi = nx.Graph(subgraph)  # 转换为有向图求解
        sub_components = nx.connected_components(subgraph_bi)
        sub_components = [item for item in sub_components]
        if len(sub_components) == 1 and len(sub_components[0]) == len(comp):
            res.append(sub_components[0])
    return res


def get_random_graphs(graph, l_list, target, multi=False):
    # 好像多进程版本并没有太多效果
    print("produce random graphs, target {}".format(target))
    res = list()
    if multi:
        pool = mtp.Pool(processes=5)
        for i in range(target):
            size = random.choice(l_list)
            res.append(pool.apply_async(
                get_single_random_graph_nodes, args=(graph, size, i)))
        pool.close()
        pool.join()
        return [item.get() for item in res]
    else:
        for i in range(target):
            size = random.choice(l_list)
            res.append(get_single_random_graph_nodes(graph, size, i))
        return res


# 这种随机化结果产生的区分度过强，看有没有其他随机的方案
def get_single_random_graph_nodes(graph, size, index):
    print("random graph index {}".format(index))
    # 注意随机游走的时候将有向图当成无向图处理
    all_nodes = list(graph.nodes.keys())  # 按照权重取值
    all_node_weights = [graph.degree(node) for node in all_nodes]
    beginer = random.choices(all_nodes, weights=all_node_weights, k=1)[0]
    # 按照权重选取下一个点
    node_set = set([beginer])
    neighbor_lists = list(set(graph.successors(beginer)) | set(
        graph.predecessors(beginer)))  # 生成随机图的时候有向图当成无向图处理
    neighbor_weights = [1 for i in range(len(neighbor_lists))]
    max_weight = 1
    while len(node_set) < size and len(neighbor_lists):
        next_node = random.choices(
            neighbor_lists, weights=neighbor_weights, k=1)[0]
        node_index = neighbor_lists.index(next_node)
        neighbor_lists.pop(node_index)
        the_weight = neighbor_weights.pop(node_index)
        max_weight = max(max_weight, the_weight)

        if (the_weight == 1 and max_weight >= 100) or (the_weight == 10 and max_weight >= 10000):  # 密集子图之后不应该再出现低权重图
            continue

        node_set.add(next_node)
        neis = list(set(graph.successors(beginer)) | set(
            graph.predecessors(beginer)))  # 区分有向图和无向图
        for nei in neis:
            if nei not in node_set:
                if nei in neighbor_lists:
                    nei_index = neighbor_lists.index(nei)
                    neighbor_weights[nei_index] *= 10  # 强调
                else:
                    neighbor_lists.append(nei)
                    neighbor_weights.append(1)
    sub_graph = nx.subgraph(graph, node_set)  # 最后还需要在子图里面去除1/4的度小的节点
    items = [(node, sub_graph.degree(node)) for node in node_set]
    remove_num_direct = min(len(node_set)//4, 6)  # 也不能去除太多了，最多去除6个

    degrees = [sub_graph.degree(node) for node in node_set]
    meandegree = (sum(degrees)/len(degrees))
    sitems = sorted(items, key=lambda i: i[1])
    res = set()
    for item in sitems[remove_num_direct:]:
        if item[1] > int(meandegree/2):  # 按照平均度数再减去一部分
            res.add(item[0])
    return res


def read_complexes(path):
    res = list()
    with open(path, 'r')as f:
        for line in f:
            line_splited = None
            if '\t' in line:
                line_splited = line.strip().split('\t')
            elif ' ' in line:
                line_splited = line.strip().split(' ')
            else:
                pass
            res.append(set(line_splited))
    return res


# 具体怎么做以后需要改进，子图合并操
def merged_data(items):
    all_merged_res = []
    for item in items:
        cur_merge_target = []
        tempres = item
        for index, single_res in enumerate(all_merged_res):
            if len(item & single_res)/(len(item | single_res)) > 0.8:
                cur_merge_target.append(index)
                tempres = tempres | single_res
        for removeindex in cur_merge_target[::-1]:
            all_merged_res.pop(removeindex)  # 从后面往前面剔除
        all_merged_res.append(tempres)
    res = list()
    for data in all_merged_res:
        if len(data) >= 2:
            res.append(data)
    return res


def complex_score(complexes, benchs):
    af_matrix = [[0 for j in range(len(benchs))]for i in range(
        len(complexes))]
    for i in range(len(complexes)):
        for j in range(len(benchs)):
            comp, targ = complexes[i], benchs[j]
            af_matrix[i][j] = pow(len(comp & targ), 2)/(len(comp)*len(targ))
    return [max(af_matrix[index]) for index, comp in enumerate(complexes)]


def savesubgraphspicture(graph, nodelists, path):
    for index, nodes in enumerate(nodelists):
        subgraph = nx.subgraph(graph, nodes)
        nx.draw(subgraph)
        plt.savefig(path+"_{}".format(index))
        plt.close()


def get_singlegraph(biggraph, item, direct, index):
    print('processing {}'.format(index))
    subgraph = biggraph.subgraph(item['complexes'])
    return single_data(subgraph, direct, item['label'], item['score'])


def construct_dgl_graph(graph):
    res = dgl.DGLGraph()
    nodes = {name: index for index, name in enumerate(graph.nodes)}

    def feat_distrubute(data, names):
        res = {}
        for name in names.keys():
            res[name] = torch.tensor(
                data[names[name][0]:names[name][1]], dtype=torch.float32).reshape(1, -1)
            res['feat'] = torch.tensor(
                data, dtype=torch.float32).reshape(1, -1)
        return res
    for node in nodes.keys():
        feat = feat_distrubute(graph.nodes[node]['w'], graph.nodes[node]['n'])
        res.add_nodes(1, feat)
    for v0, v1 in graph.edges:
        feat = feat_distrubute(graph[v0][v1]['w'], graph[v0][v1]['n'])
        res.add_edges(nodes[v0], nodes[v1], feat)
    return res


class single_data:
    def __init__(self, nxgraph, direct, label=None, score=None):
        self.label = label
        self.score = score
        self.graph = construct_dgl_graph(nxgraph)
        self.feat = self.get_default_feature(nxgraph, direct)

    def get_default_feature(self, graph: nx.Graph, direct):
        result = []
        result.append(len(graph.nodes))
        result.append(nx.density(graph))
        degrees = nx.degree(graph)
        degrees = np.array([item[1]for item in degrees])
        clusters = nx.clustering(graph)
        clusters = np.array([clusters[item] for item in clusters.keys()])
        # topologic = nx.topological_sort(graph)
        result.append(degrees.mean())
        result.append(degrees.max())
        result.append(degrees.min())
        result.append(degrees.var())

        result.append(clusters.mean())
        result.append(clusters.max())
        result.append(clusters.var())
        if direct:
            # 计算有方向的时候的补充特征
            pass
        else:
            # 计算无向的时候的补充特征
            correlation = nx.degree_pearson_correlation_coefficient(
                graph)
            result.append(correlation if correlation is not np.nan else 0.0)
        return list(result)


def get_global_nxgraph(node_path, edge_path, direct):
    print("reading graph")
    # 获取图数据
    nodes, nodematrix, nodefeatnames = read_feat_datas(node_path)
    edges, edgematrix, edgefeatnames = read_feat_datas(edge_path)
    edges = [list(item.split(' ')) for item in edges]
    nx_graph = nx.DiGraph()
    for index, item in enumerate(nodematrix):
        nx_graph.add_node(nodes[index], w=item, n=nodefeatnames)
    if direct:
        for index, item in enumerate(edgematrix):
            nx_graph.add_edge(edges[index][0], edges[index]
                              [1], w=item, n=edgefeatnames)
    else:
        for index, item in enumerate(edgematrix):  # 无向图可以这么处理，重复
            nx_graph.add_edge(edges[index][0], edges[index]
                              [1], w=item, n=edgefeatnames)
            nx_graph.add_edge(edges[index][1], edges[index]
                              [0], w=item, n=edgefeatnames)
    return nx_graph


def path_process(graphname, benchname, refername, basedir, direct):
    train_datasets_path = basedir + "datasets/train_datasets/{}_{}_{}_{}".format(
        benchname, graphname, refername, "D" if direct else "U")

    refer_results_path = basedir + "datasets/refer_results/{}_{}".format(
        graphname,  refername)
    refer_results_expand_path = refer_results_path+'_expand'
    select_datasets_path = basedir + \
        "datasets/select_datasets/{}_{}_{}".format(
            graphname, refername, "D" if direct else "U")
    select_datasets_expand_path = select_datasets_path+'_expand'

    pictures_dir = basedir + "datasets/pictures/"
    bench_path = basedir + "bench/{}/complexes".format(benchname)
    edges_path = basedir + "network/{}/edges".format(graphname)
    nodesfeat_path = basedir + \
        "network/{}/nodes_feat_final".format(graphname)
    edgesfeat_path = basedir + \
        "network/{}/edges_feat_final".format(graphname)
    return train_datasets_path, refer_results_path, refer_results_expand_path, select_datasets_path, select_datasets_expand_path, pictures_dir, bench_path, edges_path, nodesfeat_path, edgesfeat_path


def add_labels_on_datas(origin_bench_data, subgraphed_bench_data, refer_data, random_data):
    bench_data = [[comp, 1.0] for comp in subgraphed_bench_data]

    refer_scores = complex_score(refer_data, origin_bench_data)
    refer_data_neg, refer_data_pos = [], []
    for index, comp in enumerate(refer_data):
        if refer_scores[index] <= 0.20:
            refer_data_neg.append([comp, refer_scores[index]])
        elif refer_scores[index] > 0.40:
            refer_data_pos.append([comp, refer_scores[index]])
        else:
            pass

    random.shuffle(refer_data)
    random_scores = complex_score(random_data, origin_bench_data)
    random_data_neg, random_data_pos = [], []
    for index, comp in enumerate(random_data):
        if random_scores[index] <= 0.20:
            random_data_neg.append([comp, random_scores[index]])
        elif random_scores[index] > 0.40:
            random_data_pos.append([comp, random_scores[index]])
        else:
            pass

    all_datas = [bench_data, random_data_neg, refer_data_pos, refer_data_neg]

    res = [[{'complexes': item[0], 'label': index, 'score': item[1]}
            for item in one_class_data] for index, one_class_data in enumerate(all_datas)]
    return res


def construct_and_storation_subgraphs(path, graph, subs, direct):
    storation_size = 500
    begin_index = 0
    datasets = []
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    while begin_index < len(subs):
        next_index = min(begin_index+storation_size, len(subs))
        cur_refer_data = subs[begin_index: next_index]
        cur_datasets = [get_singlegraph(
            graph, item, direct, index+begin_index) for index, item in enumerate(cur_refer_data)]
        datasets.extend(cur_datasets)
        with open(path+'/'+str(begin_index), 'wb') as f:
            pickle.dump(cur_datasets, f)
        begin_index = next_index
    return datasets


def reload_subgraphs(path):
    file_names = list(map(str, sorted(map(int, os.listdir(path)))))
    result = []
    for file_name in file_names:
        with open(path+'/'+file_name, 'rb')as f:
            result.extend(pickle.load(f))
    return result


def trainmodel_datasets(recompute=False, direct=False, graphname="DIP", benchname="CYC2008", refername="coach", basedir=""):
    train_datasets_path, refer_results_path, refer_results_expand_path, select_datasets_path, select_datasets_expand_path, pictures_dir, bench_path, edges_path, nodesfeat_path, edgesfeat_path = path_process(
        graphname, benchname, refername, basedir, direct)
    if not recompute and os.path.exists(train_datasets_path):
        reloaded_datas = []
        for names in sorted(os.listdir(train_datasets_path)):
            data = list(reload_subgraphs(
                train_datasets_path+"/{}".format(name)) for name in sorted(names))
            reloaded_datas.append(data[0])
        return reloaded_datas, os.path.basename(train_datasets_path)

    # refer方法的图跑出来
    refermethods.main(method_name=refername, edges_path=edges_path,
                      result_path=refer_results_path, expand=False, basedir=basedir+"refer", recompute=recompute)
    nx_graph = get_global_nxgraph(nodesfeat_path, edgesfeat_path, direct)
    # dgl_graph = single_data(nx_graph, direct).graph
    origin_bench_data = read_complexes(bench_path)
    # 接下来需要提取真正的graph，找出所有的subgraph，因为有些点是不存在的
    subgraphed_bench_data = subgraphs(origin_bench_data, nx_graph)
    refer_data = read_complexes(refer_results_path)
    random_data = get_random_graphs(
        nx_graph, [len(item) for item in origin_bench_data], min(1000, len(subgraphed_bench_data)+len(refer_data)), multi=False)

    # 接下来归并处理
    labeled_datas = add_labels_on_datas(
        origin_bench_data, subgraphed_bench_data, refer_data, random_data)
    construct_graph_datas = [construct_and_storation_subgraphs(
        train_datasets_path+"/{}".format(index), nx_graph, single_class, direct) for index, single_class in enumerate(labeled_datas)]
    return construct_graph_datas, os.path.basename(train_datasets_path)


def selectcomplex_datasets(recompute=False, direct=False, graphname="DIP", benchname="CYC2008", refername="coach", basedir=""):
    train_datasets_path, refer_results_path, refer_results_expand_path, select_datasets_path, select_datasets_expand_path, pictures_dir, bench_path, edges_path, nodesfeat_path, edgesfeat_path = path_process(
        graphname, benchname, refername, basedir, direct)

    # refer方法的图跑出来
    refer_data = refermethods.main(method_name=refername, edges_path=edges_path,
                                   result_path=refer_results_path, expand=False, basedir=basedir+"refer", recompute=recompute)
    expand_refer_data = refermethods.main(method_name=refername, edges_path=edges_path,
                                          result_path=refer_results_expand_path, expand=True, basedir=basedir+"refer", recompute=recompute)

    if not recompute and os.path.exists(select_datasets_expand_path):
        result = reload_subgraphs(select_datasets_expand_path)
        return refer_data, expand_refer_data, result

    nx_graph = get_global_nxgraph(nodesfeat_path, edgesfeat_path, direct)
    bench_data = read_complexes(bench_path)
    scores = complex_score(expand_refer_data, bench_data)
    expand_refer_data_with_label = [{'complexes': item, 'label': 2 if scores[index] >= 0.25 else 3, 'score': scores[index]}
                                    for index, item in enumerate(expand_refer_data)]
    datasets = construct_and_storation_subgraphs(
        select_datasets_expand_path, nx_graph, expand_refer_data_with_label, direct)
    return refer_data, expand_refer_data, datasets


if __name__ == "__main__":
    # trainmodel_datasets(recompute=True)
    # selectcomplex_datasets()
    pass
