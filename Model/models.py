from torch import nn as torchnn
import torch
import dgl
import dgl.nn as dglnn


class DGLInit(torchnn.Module):
    def __init__(self, innode_size, inedge_size, hidden_size):
        super().__init__()
        self.init_weight_node = torchnn.Linear(
            innode_size, hidden_size, bias=True)
        self.init_weight_edge = torchnn.Linear(
            inedge_size, hidden_size, bias=True)
        self.activate = lambda x: x

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.ndata['h'] = self.activate(self.init_weight_node(
            dgl_data.ndata['feat']))
        dgl_data.edata['h'] = self.activate(self.init_weight_edge(
            dgl_data.edata['feat']))
        return dgl_data


class DGLToRandom(torchnn.Module):
    def __init__(self, innode_size, inedge_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, dgl_data: dgl.DGLGraph):
        temp_node_feat = dgl_data.ndata['feat']
        temp_edge_feat = dgl_data.edata['feat']
        dgl_data.ndata['h'] = torch.rand(
            (temp_node_feat.size()[0], self.hidden_size))
        dgl_data.edata['h'] = torch.rand(
            (temp_edge_feat.size()[0], self.hidden_size))
        return dgl_data


class Node_with_edge_GCN__(torchnn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.edgeflowweight = torchnn.Linear(
            hidden_size, hidden_size, bias=True)
        self.edgesrcweight = torchnn.Linear(
            hidden_size, hidden_size, bias=True)
        self.edgefinaleweight = torchnn.Linear(
            hidden_size*2, hidden_size, bias=True)
        self.nodeweight = torchnn.Linear(
            hidden_size, hidden_size, bias=True)
        # self.batchnorm = torch.nn.BatchNorm1d(hidden_size)
        self.activate = torchnn.LeakyReLU()

    def msg_gcn(self, edge):
        # 先是点汇聚边信息，然后更新边信息
        old_edge_data = edge.data['h']
        nodeflowfeat = self.edgeflowweight(
            edge.dst['h'] - edge.src['h'])+self.edgesrcweight(edge.src['h'])

        new_edgefeat = self.activate(self.edgefinaleweight(
            torch.cat([edge.data['h'], nodeflowfeat], -1)))
        edge.data['h'] = new_edgefeat
        return {'msg': old_edge_data}

    def reduce_gcn(self, node):
        reduce = torch.mean(node.mailbox['msg'], 1)  # 箭头指向某一个节点。所有的箭头的信息汇聚到一起
        return {'reduce': reduce}

    def apply_gcn(self, node):
        new_nodefeat = self.activate(self.nodeweight(node.data['reduce']))
        return {'h': new_nodefeat}

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.update_all(self.msg_gcn, self.reduce_gcn, self.apply_gcn)
        return dgl_data


class Only_Node_GCN__(torchnn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.nodeweight = torchnn.Linear(
            hidden_size, hidden_size, bias=True)
        self.activate = torchnn.LeakyReLU()

    def msg_gcn(self, edge):
        return {'msg': edge.src['h']}

    def reduce_gcn(self, node):
        catnodefeat = torch.cat(
            (node.mailbox['msg'], torch.unsqueeze(node.data['h'], 1)), 1)
        meannodes = torch.mean(catnodefeat, 1)
        return {'reduce': meannodes}

    def apply_gcn(self, node):
        new_nodefeat = self.activate(self.nodeweight(node.data['reduce']))
        return {'h': new_nodefeat}

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.update_all(self.msg_gcn, self.reduce_gcn, self.apply_gcn)
        return dgl_data


class Only_Edge_GCN__(torchnn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.edgeflowweight = torchnn.Linear(
            hidden_size, hidden_size, bias=True)
        self.edgesrcweight = torchnn.Linear(
            hidden_size, hidden_size, bias=True)
        self.activate = torchnn.LeakyReLU()

    def msg_gcn(self, edge):
        nodeflowfeat = self.edgeflowweight(
            edge.dst['h'] - edge.src['h'])+self.edgesrcweight(edge.src['h'])
        return {'msg': nodeflowfeat}

    def reduce_gcn(self, node):
        # 箭头指向某一个节点。所有的箭头的信息汇聚到一起
        meannode = torch.mean(node.mailbox['msg'], 1)
        return {'reduce': meannode}

    def apply_gcn(self, node):
        nodestack = torch.stack((node.data['reduce'], node.data['h']), 2)
        nodemax, _ = torch.max(nodestack, 2)
        return {'h': nodemax}

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.update_all(self.msg_gcn, self.reduce_gcn, self.apply_gcn)
        return dgl_data


class Edge_to_node__(torchnn.Module):
    def __init__(self):
        super().__init__()

    def msg_gcn(self, edge):
        # 先是点汇聚边信息，然后更新边信息
        return {'msg': edge.data['feat']}

    def reduce_gcn(self, node):
        reduce = torch.mean(node.mailbox['msg'], 1)  # 箭头指向某一个节点。所有的箭头的信息汇聚到一起
        return {'reduce': reduce}

    def apply_gcn(self, node):
        return {'h': node.data['reduce']}

    def forward(self, dgl_data: dgl.DGLGraph):
        dgl_data.update_all(self.msg_gcn, self.reduce_gcn, self.apply_gcn)
        return dgl_data


class GCN_process(torchnn.Module):
    def __init__(self, hidden_size, layer_num, singlegcn):
        super().__init__()
        self.GCNlayers = torchnn.ModuleList()
        for _ in range(layer_num):
            self.GCNlayers.append(singlegcn(hidden_size))

    def forward(self, dgl_data):
        for singlelayer in self.GCNlayers:
            dgl_data = singlelayer(dgl_data)
        return dgl_data


class GCN_readout_cat(torchnn.Module):
    def __init__(self, hidden_size, node=True, edge=True):
        super().__init__()
        self.getnode = node
        self.getedge = edge
        self.weight_node = torchnn.Linear(
            hidden_size*2*(int(node+edge)), hidden_size, bias=True)
        self.activate = torchnn.LeakyReLU()

    def forward(self, dgl_data):
        if self.getnode and self.getedge:
            dgl_feat = torch.cat([
                dgl.mean_nodes(dgl_data, 'h'),
                dgl.max_nodes(dgl_data, 'h'),
                dgl.mean_edges(dgl_data, 'h'),
                dgl.max_edges(dgl_data, 'h'),
            ], -1)
        elif self.getnode:
            dgl_feat = torch.cat([
                dgl.mean_nodes(dgl_data, 'h'),
                dgl.max_nodes(dgl_data, 'h')], -1)
        else:
            dgl_feat = torch.cat([
                dgl.mean_edges(dgl_data, 'h'),
                dgl.max_edges(dgl_data, 'h')], -1)
        dgl_predict = self.activate(self.weight_node(dgl_feat))
        return dgl_predict


class GCN_readout_max(torchnn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight_node = torchnn.Linear(
            hidden_size*4, hidden_size, bias=True)
        self.activate = torchnn.LeakyReLU()

    def forward(self, dgl_data):
        dgl_feat, _ = torch.max(torch.stack([
            dgl.mean_nodes(dgl_data, 'h'),
            dgl.max_nodes(dgl_data, 'h'),
            dgl.mean_edges(dgl_data, 'h'),
            dgl.max_edges(dgl_data, 'h'),
        ], 2), -1)
        return dgl_feat


class GCN_readout_topk(torchnn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.readoutmax = GCN_readout_max(hidden_size)
        self.activate = torchnn.LeakyReLU()
        self.weight = torchnn.Linear(hidden_size*4, hidden_size)

    def forward(self, dgl_data):
        topknodes, _ = dgl.topk_nodes(dgl_data, 'h', 4)
        topkedges, _ = dgl.topk_edges(dgl_data, 'h', 4)
        meannode = torch.mean(topknodes, 1)
        maxnode, _ = torch.max(topknodes, 1)
        meanedge = torch.mean(topkedges, 1)
        maxedge, _ = torch.max(topkedges, 1)
        dgl_feat = torch.cat([meannode, maxnode, meanedge, maxedge], -1)
        dgl_predict = self.activate(self.weight(dgl_feat))
        return dgl_predict


class Result_predictor(torchnn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.class_weights_0 = torchnn.Linear(input_size, hidden_size)
        self.class_weights_1 = torchnn.Linear(hidden_size, output_size)
        self.score_weights_0 = torchnn.Linear(input_size, hidden_size)
        self.score_weights_1 = torchnn.Linear(hidden_size, 1)
        self.activate = torchnn.LeakyReLU()

    def forward(self, input_data):
        class_middle = self.activate(self.class_weights_0(input_data))
        class_res = self.class_weights_1(class_middle)
        score_middle = self.activate(self.score_weights_0(input_data))
        score_res = self.score_weights_1(score_middle)
        score_res = torchnn.Sigmoid()(score_res)  # 由于后续的逻辑回归损失，以及需要结果在0~1，这里接上sigmoid
        return class_res, score_res


class Only_struct_GCN(torchnn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "Only_struct_GCN"
        hidden_size = args.hidden_size
        self.nodeedge_feat_init = DGLToRandom(
            args.nodefeatsize, args.edgefeatsize, hidden_size)
        self.gcn_process = GCN_process(
            hidden_size, args.gcn_layers, Only_Node_GCN__)
        self.gcn_predict = GCN_readout_cat(
            hidden_size, node=True, edge=True)
        self.predict_result = Result_predictor(
            hidden_size, hidden_size, args.output_size)

    def forward(self, dgl_data, base_data):
        dgl_data_init = self.nodeedge_feat_init(dgl_data)
        dgl_data_gcn = self.gcn_process(dgl_data_init)
        dgl_data_feat = self.gcn_predict(dgl_data_gcn)
        return self.predict_result(dgl_data_feat)


class Only_Node_feat_GCN(torchnn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "Only_Node_feat_GCN"
        hidden_size = args.hidden_size
        self.nodeedge_feat_init = DGLInit(
            args.nodefeatsize, args.edgefeatsize, hidden_size)
        self.gcn_process = GCN_process(
            hidden_size, args.gcn_layers, Only_Node_GCN__)
        self.gcn_predict = GCN_readout_cat(
            hidden_size, node=True, edge=False)
        self.predict_result = Result_predictor(
            hidden_size, hidden_size, args.output_size)

    def forward(self, dgl_data, base_data):
        dgl_data_init = self.nodeedge_feat_init(dgl_data)
        dgl_data_gcn = self.gcn_process(dgl_data_init)
        dgl_data_feat = self.gcn_predict(dgl_data_gcn)
        return self.predict_result(dgl_data_feat)


class Only_Edge_feat_GCN(torchnn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "Only_Edge_feat_GCN"
        hidden_size = args.hidden_size
        self.nodeedge_feat_init = DGLInit(
            args.nodefeatsize, args.edgefeatsize, hidden_size)
        self.edgetonode = Edge_to_node__()
        self.gcn_process = GCN_process(
            hidden_size, args.gcn_layers, Only_Edge_GCN__)
        self.gcn_predict = GCN_readout_cat(
            hidden_size, node=True, edge=True)
        self.predict_result = Result_predictor(
            hidden_size, hidden_size, args.output_size)

    def forward(self, dgl_data, base_data):
        dgl_data = self.edgetonode(dgl_data)  # 使用edge重新改编node
        dgl_data_init = self.nodeedge_feat_init(dgl_data)
        dgl_data_gcn = self.gcn_process(dgl_data_init)
        dgl_data_feat = self.gcn_predict(dgl_data_gcn)
        return self.predict_result(dgl_data_feat)


class Fusion_GCN(torchnn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "Fusion_GCN"
        hidden_size = args.hidden_size
        self.nodeedge_feat_init = DGLInit(
            args.nodefeatsize, args.edgefeatsize, hidden_size)

        self.gcn_process = GCN_process(
            hidden_size, args.gcn_layers, Node_with_edge_GCN__)
        self.gcn_predict = GCN_readout_cat(hidden_size, node=True, edge=True)
        self.predict_result = Result_predictor(
            hidden_size, hidden_size, args.output_size)

    def forward(self, dgl_data, base_data):
        dgl_data_init = self.nodeedge_feat_init(dgl_data)
        dgl_data_gcn = self.gcn_process(dgl_data_init)
        dgl_data_feat = self.gcn_predict(dgl_data_gcn)
        return self.predict_result(dgl_data_feat)


class Only_Topologi(torchnn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "gcnwithtopo"
        hidden_size = args.hidden_size
        self.base_feat_init = torchnn.Linear(
            args.graphfeatsize, hidden_size, bias=True)
        self.predict_result = Result_predictor(
            hidden_size, hidden_size, args.output_size)

    def forward(self, dgl_data, base_data):
        base_feat = self.base_feat_init(base_data)
        return self.predict_result(base_feat)


class Fusion_GCN_with_Topologi(torchnn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "gcnwithtopo"
        hidden_size = args.hidden_size
        self.nodeedge_feat_init = DGLInit(
            args.nodefeatsize, args.edgefeatsize, hidden_size)
        self.base_feat_init = torchnn.Linear(
            args.graphfeatsize, hidden_size, bias=True)

        self.gcn_process = GCN_process(
            hidden_size, args.gcn_layers, Node_with_edge_GCN__)
        self.gcn_predict = GCN_readout_cat(hidden_size, node=True, edge=True)
        self.predict_result = Result_predictor(
            hidden_size*2, hidden_size, args.output_size)

    def forward(self, dgl_data, base_data):
        dgl_data_init = self.nodeedge_feat_init(dgl_data)
        dgl_data_gcn = self.gcn_process(dgl_data_init)
        dgl_data_feat = self.gcn_predict(dgl_data_gcn)
        base_feat = self.base_feat_init(base_data)
        final_feat = torch.cat([dgl_data_feat, base_feat], -1)
        return self.predict_result(final_feat)


class GCN_with_Topologi_with_topk(torchnn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "gcnwithtopo"
        hidden_size = args.hidden_size
        self.nodeedge_feat_init = DGLInit(
            args.nodefeatsize, args.edgefeatsize, hidden_size)
        self.base_feat_init = torchnn.Linear(
            args.graphfeatsize, hidden_size, bias=True)

        self.gcn_process = GCN_process(
            hidden_size, args.gcn_layers, Node_with_edge_GCN__)
        self.gcn_predict = GCN_readout_topk(hidden_size)
        self.predict_result = Result_predictor(
            hidden_size*2, hidden_size, args.output_size)

    def forward(self, dgl_data, base_data):
        dgl_data_init = self.nodeedge_feat_init(dgl_data)
        dgl_data_gcn = self.gcn_process(dgl_data_init)
        dgl_data_feat = self.gcn_predict(dgl_data_gcn)
        base_feat = self.base_feat_init(base_data)
        final_feat = torch.cat([dgl_data_feat, base_feat], -1)
        return self.predict_result(final_feat)


class GCN_with_Topologi_with_max(torchnn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = "gcnwithtopo"
        hidden_size = args.hidden_size
        self.nodeedge_feat_init = DGLInit(
            args.nodefeatsize, args.edgefeatsize, hidden_size)
        self.base_feat_init = torchnn.Linear(
            args.graphfeatsize, hidden_size, bias=True)

        self.gcn_process = GCN_process(
            hidden_size, args.gcn_layers, Node_with_edge_GCN__)
        self.gcn_predict = GCN_readout_max(hidden_size)
        self.predict_result = Result_predictor(
            hidden_size*2, hidden_size, args.output_size)

    def forward(self, dgl_data, base_data):
        dgl_data_init = self.nodeedge_feat_init(dgl_data)
        dgl_data_gcn = self.gcn_process(dgl_data_init)
        dgl_data_feat = self.gcn_predict(dgl_data_gcn)
        base_feat = self.base_feat_init(base_data)
        final_feat = torch.cat([dgl_data_feat, base_feat], -1)
        return self.predict_result(final_feat)


def get_model(args):
    if args.modelname == "Only_struct_GCN":
        return Only_struct_GCN(args)
    if args.modelname == "Only_Node_feat_GCN":
        return Only_Node_feat_GCN(args)
    if args.modelname == "Only_Edge_feat_GCN":
        return Only_Edge_feat_GCN(args)
    if args.modelname == "Fusion_GCN":
        return Fusion_GCN(args)

    if args.modelname == "Only_Topologi":
        return Only_Topologi(args)
    if args.modelname == "Fusion_GCN_with_Topologi":
        return Fusion_GCN_with_Topologi(args)

    if args.modelname == "GCN_with_Topologi_with_topk":
        return GCN_with_Topologi_with_topk(args)
    if args.modelname == "GCN_with_Topologi_with_max":
        return GCN_with_Topologi_with_max(args)


if __name__ == '__main__':
    pass
