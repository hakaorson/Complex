import collections
import numpy as np
import math


class ItemAffinity():
    def __init__(self, list_a, list_b):
        self.list_a = list_a
        self.list_b = list_b
        self.set_a = set(self.list_a)
        self.set_b = set(self.list_b)
        self.score = self.score()

    def score(self):
        NotImplemented


class NAAffinity(ItemAffinity):  # neighborhood affinity score
    def __init__(self, list_a, list_b):
        super().__init__(list_a, list_b)

    def score(self):
        set_and = self.set_a & self.set_b
        result = pow(len(set_and), 2)/(len(self.set_a)*len(self.set_b))
        return(result)


class OLAffinity(ItemAffinity):  # overlap score
    def __init__(self, list_a, list_b):
        super().__init__(list_a, list_b)

    def score(self):
        set_and = self.set_a & self.set_b
        result = len(set_and)/(len(self.set_a)*len(self.set_b))
        return(result)


class CoocAffinity(ItemAffinity):  # co-occurrence score,共同蛋白质数量
    def __init__(self, list_a, list_b):
        super().__init__(list_a, list_b)

    def score(self):
        set_and = self.set_a & self.set_b
        result = len(set_and)
        return(result)


class ClusterQuality():
    def __init__(self, cluster_bench, cluster_predict, affinity_method):
        self.cluster_bench = cluster_bench
        self.cluster_predict = cluster_predict
        self.id_map = self.get_id_map()
        self.affinity_method = affinity_method
        self.affinity_matrix = self.get_affinity_matrix()

    def get_id_map(self):
        id_map = collections.defaultdict(int)
        all_item = set([item for cluster in self.cluster_bench for item in cluster]) | \
            set([item for cluster in self.cluster_predict for item in cluster])
        for index, item in enumerate(all_item):
            id_map[item] = index
        return(id_map)

    def get_affinity_matrix(self):
        af_matrix = [[0 for j in range(len(self.cluster_predict))]for i in range(
            len(self.cluster_bench))]
        for i in range(len(self.cluster_bench)):
            for j in range(len(self.cluster_predict)):
                ij_affinity = self.affinity_method(
                    self.cluster_bench[i], self.cluster_predict[j])
                af_matrix[i][j] = ij_affinity.score
        return af_matrix

    def score(self):
        NotImplemented


class ClusterQualityF1_MMR(ClusterQuality):
    def __init__(self, cluster_bench, cluster_predict, affinity_method=None, threshold=None):
        super().__init__(cluster_bench, cluster_predict, affinity_method)
        self.threshold = threshold

    def score(self):
        np_matrix = np.array(self.affinity_matrix)

        mmrscores = np.mean(np.max(np_matrix, 1))

        bool_matrix = np_matrix >= self.threshold
        prec_matrix = np.sum(bool_matrix, 0) > 0  # 这个计算是不是有问题
        reca_matrix = np.sum(bool_matrix, 1) > 0
        precision = sum(prec_matrix)/len(prec_matrix)
        recall = sum(reca_matrix)/len(reca_matrix)
        f1 = 2*precision*recall/(precision+recall)
        return(precision, recall, f1, mmrscores)


class ClusterQualitySN_PPV_Acc(ClusterQuality):
    def __init__(self, cluster_bench, cluster_predict, affinity_method=None, threshold=None):
        super().__init__(cluster_bench, cluster_predict, CoocAffinity)
        self.threshold = threshold

    def score(self):
        np_matrix = np.array(self.affinity_matrix)  # bench*predict
        bench_matchedmax = np.max(np_matrix, 1)
        sum_bench = sum([len(item) for item in self.cluster_bench])
        sum_bench_matched = int(sum(bench_matchedmax))
        sn_score = sum_bench_matched/sum_bench  # 每一个标准复合物各自最多被找到多少

        predict_matchedmax = np.max(np_matrix, 1)
        sum_predict_matched = int(sum(predict_matchedmax))
        ppv_score = sum_predict_matched / \
            np.sum(np_matrix)  # 每一个预测出来的复合物是不是具有特定性
        acc_score = math.sqrt(sn_score*ppv_score)
        return sn_score, ppv_score, acc_score
