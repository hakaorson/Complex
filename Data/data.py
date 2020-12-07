import os
import pickle
import queue
import random
import subprocess
from multiprocessing import pool as mtp

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing

from Data.refer import runinpy3


def get_datas(data_path):
    ids, datas = [], []
    with open(data_path, 'r') as f:
        next(f)
        for item in f:
            item_splited = item.strip().split('\t')
            ids.append(item_splited[0])
            datas.append(list(map(float, item_splited[1:])))
    return ids, datas


# 数据预处理，一些归一化等等
def dataprocess(matrix):
    matrix = np.array(matrix)
    matrix = preprocessing.normalize(matrix, axis=0)
    return matrix


def subgraphs(complexes, graph):
    res = []
    all_complex_nodes = set()
    all_graph_nodes = set(graph.nodes)
    for comp in complexes:
        all_complex_nodes = all_complex_nodes | comp
        subgraph = nx.subgraph(graph, comp)
        subgraph_bi = nx.Graph(subgraph)  # 转换为有向图求解
        sub_components = nx.connected_components(subgraph_bi)
        for sub_component in sub_components:
            res.append(sub_component)
    print("has {} protein not in graph".format(
        len(all_complex_nodes-all_graph_nodes)))
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


# 去重处理
# 去重处理需要按照score NA的0.25作为分界
def remove_duplicate(complexes, targets):
    af_matrix = [[0 for j in range(len(targets))]for i in range(
        len(complexes))]
    for i in range(len(complexes)):
        for j in range(len(targets)):
            comp, targ = complexes[i], targets[j]
            af_matrix[i][j] = pow(len(comp & targ), 2)/(len(comp)*len(targ))
    res = []
    for index, comp in enumerate(complexes):
        if max(af_matrix[index]) >= 0.25:
            continue
        res.append(comp)
    return res


def savesubgraphs(graph, nodelists, path):
    for index, nodes in enumerate(nodelists):
        subgraph = nx.subgraph(graph, nodes)
        nx.draw(subgraph)
        plt.savefig(path+"_{}".format(index))
        plt.close()


def get_singlegraph(biggraph, nodes, direct, label, index):
    print('processing {}'.format(index))
    subgraph = biggraph.subgraph(nodes)
    return single_data(subgraph, direct, label)


class single_data:
    def __init__(self, graph, direct, label=None):
        self.label = label
        self.graph = self.dgl_graph(graph)
        self.feat = self.get_default_feature(graph, direct)

    def dgl_graph(self, graph: nx.Graph):
        res = dgl.DGLGraph()
        nodes = list(graph.nodes)
        for index, node in enumerate(nodes):
            data = torch.tensor(
                graph.nodes[node]['w'], dtype=torch.float32).reshape(1, -1)
            deg = torch.tensor(graph.degree(
                node), dtype=torch.float32).reshape(1, -1)
            res.add_nodes(1, {'feat': data, 'degree': deg})
            # res.add_edge(index, index)
        for v0, v1 in graph.edges:
            data = torch.tensor(
                graph[v0][v1]['w'], dtype=torch.float32).reshape(1, -1)
            res.add_edges(nodes.index(v0), nodes.index(v1), {'feat': data})
        return res

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
    nodes, nodematrix = get_datas(node_path)
    edges, edgematrix = get_datas(edge_path)
    edges = [list(item.split(' ')) for item in edges]
    # 归一化处理
    # nodematrix = dataprocess(nodematrix)
    # edgematrix = dataprocess(edgematrix)
    nx_graph = nx.DiGraph()
    for index, item in enumerate(nodematrix):
        nx_graph.add_node(nodes[index], w=item)
    if direct:
        for index, item in enumerate(edgematrix):
            nx_graph.add_edge(edges[index][0], edges[index][1], w=item)
    else:
        for index, item in enumerate(edgematrix):  # 无向图可以这么处理，重复
            nx_graph.add_edge(edges[index][0], edges[index][1], w=item)
            nx_graph.add_edge(edges[index][1], edges[index][0], w=item)
    return nx_graph


def path_process(graphname, benchname, refername, basedir):
    train_datasets_path = basedir + "datasets/train_datasets/{}_{}_{}".format(
        graphname, benchname, refername)

    refer_results_path = basedir + "datasets/refer_results/{}_{}".format(
        graphname,  refername)
    refer_results_expand_path = refer_results_path+'_expand'
    select_datasets_path = basedir + \
        "datasets/select_datasets/{}_{}".format(graphname, refername)
    select_datasets_expand_path = select_datasets_path+'_expand'

    pictures_dir = basedir + "datasets/pictures/"
    bench_path = basedir + "bench/{}/complexes".format(benchname)
    edges_path = basedir + "network/{}/edges".format(graphname)
    nodesfeat_path = basedir + \
        "network/{}/nodes_feat_processed".format(graphname)
    edgesfeat_path = basedir + \
        "network/{}/edges_feat_processed".format(graphname)
    return train_datasets_path, refer_results_path, refer_results_expand_path, select_datasets_path, select_datasets_expand_path, pictures_dir, bench_path, edges_path, nodesfeat_path, edgesfeat_path


def trainmodel_datasets(recompute=False, direct=False, graphname="DIP", benchname="CYC2008", refername="coach", basedir=""):
    train_datasets_path, refer_results_path, refer_results_expand_path, select_datasets_path, select_datasets_expand_path, pictures_dir, bench_path, edges_path, nodesfeat_path, edgesfeat_path = path_process(
        graphname, benchname, refername, basedir)
    if not recompute and os.path.exists(train_datasets_path):
        with open(train_datasets_path, 'rb') as f:
            print('train_datasets exits, directly return')
            # 注意必须是在哪写的数据，就在哪解析，不能在data.py运行存储的数据，然后在first_stage解析
            return pickle.load(f)
    # bench必须存在
    assert(os.path.exists(bench_path))
    assert((os.path.exists(nodesfeat_path)
            and os.path.exists(edgesfeat_path)))  # 特征必须准备好，因为特征是对于整个图来说的，不会经常变换

    # refer方法的图跑出来
    if (not (os.path.exists(refer_results_path))) or recompute:
        runinpy3.main(method_name=refername, edges_path=edges_path,
                      result_path=refer_results_path, expand=False, basedir=basedir+"refer")
    nx_graph = get_global_nxgraph(nodesfeat_path, edgesfeat_path, direct)
    # dgl_graph = single_data(nx_graph, direct).graph
    bench_data = read_complexes(bench_path)
    middle_data = read_complexes(refer_results_path)
    random_data = get_random_graphs(
        nx_graph, [len(item) for item in bench_data + middle_data], (len(bench_data)+len(middle_data)), multi=False)  # TODO 设定随机的数目

    # 接下来需要提取真正的graph，找出所有的subgraph
    bench_data = subgraphs(bench_data, nx_graph)
    # 接下来归并处理
    bench_data = merged_data(bench_data)  # 621->555
    middle_data = merged_data(middle_data)  # 888->416
    random_data = merged_data(random_data)  # 129->99
    # 接下来去重
    middle_data = remove_duplicate(middle_data, bench_data)
    random_data = remove_duplicate(random_data, bench_data+middle_data)
    # 存储图片
    savesubgraphs(nx_graph, bench_data[:10], pictures_dir+"bench")
    savesubgraphs(nx_graph, middle_data[:10], pictures_dir+"middle")
    savesubgraphs(nx_graph, random_data[:10], pictures_dir+"random")
    # 整理成数据集
    all_datas = []
    all_datas.extend([item, 0] for item in bench_data)
    all_datas.extend([item, 1] for item in middle_data)
    all_datas.extend([item, 2] for item in random_data)
    # 多进程处理
    multi_res = []
    pool = mtp.Pool(processes=10)
    for index, (item, label) in enumerate(all_datas):
        multi_res.append(pool.apply_async(
            get_singlegraph, args=(nx_graph, item, direct, label, index)))
    pool.close()
    pool.join()
    datasets = [item.get() for item in multi_res]

    with open(train_datasets_path, 'wb') as f:
        pickle.dump(datasets, f)
    return datasets


def selectcomplex_datasets(recompute=False, direct=False, graphname="DIP", refername="coach", basedir=""):
    train_datasets_path, refer_results_path, refer_results_expand_path, select_datasets_path, select_datasets_expand_path, pictures_dir, bench_path, edges_path, nodesfeat_path, edgesfeat_path = path_process(
        graphname, "", refername, basedir)

    # refer方法的图跑出来
    if not (os.path.exists(refer_results_path)) or recompute:
        runinpy3.main(method_name=refername, edges_path=edges_path,
                      result_path=refer_results_path, expand=False, basedir=basedir+"refer")
    refer_data = read_complexes(refer_results_path)
    # refer方法的图跑出来
    if not (os.path.exists(refer_results_expand_path)) or recompute:
        runinpy3.main(method_name=refername, edges_path=edges_path,
                      result_path=refer_results_expand_path, expand=True, basedir=basedir+"refer")
    expand_refer_data = read_complexes(refer_results_expand_path)

    if not recompute and os.path.exists(select_datasets_expand_path):
        with open(select_datasets_expand_path, 'rb')as f:
            result = pickle.load(f)
        return refer_data, expand_refer_data, result
    nx_graph = get_global_nxgraph(nodesfeat_path, edgesfeat_path, direct)
    # datasets = [get_singlegraph(nx_graph, item, direct, -1)
    #             for item in candi_data]  # -1代表无意义
    # return datasets
    multi_res = []
    pool = mtp.Pool(processes=10)
    for index, item in enumerate(expand_refer_data):
        multi_res.append(pool.apply_async(
            get_singlegraph, args=(nx_graph, item, direct, -1, index)))
    pool.close()
    pool.join()
    result = [item.get() for item in multi_res]
    with open(select_datasets_expand_path, 'wb') as f:
        pickle.dump(result, f)
    return refer_data, expand_refer_data, result


if __name__ == "__main__":
    # trainmodel_datasets(recompute=True)
    # selectcomplex_datasets()
    pass
