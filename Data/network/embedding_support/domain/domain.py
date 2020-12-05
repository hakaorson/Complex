import networkx as nx


def readDat(path):
    nodes, edges = set(), set()
    with open(path, 'r') as f:
        cur_left, cur_right = '', ''
        cur_nums = 0
        for line in f:
            linelist = line.strip().split('\t')
            if len(linelist) == 0:
                continue
            if linelist[0] == r'//':
                nodes.add(cur_left)
                nodes.add(cur_right)
                edges.add(tuple([cur_left, cur_right, cur_nums]))
                cur_nums = 0
            if linelist[0] == '#=ID':
                # 需要去重lig
                cur_left, cur_right = linelist[1], linelist[2]
                if "_LIG_" in cur_left:
                    cur_left = cur_left.split("_LIG_")[0]
                if "_LIG_" in cur_right:
                    cur_right = cur_right.split("_LIG_")[0]
            if linelist[0] == '#=IF':
                size = len(linelist[1].split(' '))-1
                cur_nums += size
    return nodes, edges


def readMapping(path):
    res = {}
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            linelist = line.strip().split('\t')
            pfname = linelist[5]
            pfid = linelist[4][:7]  # 去除小数部分
            res[pfname] = pfid
    return res


if __name__ == "__main__":
    nodes, edges = readDat('origin_data/3did_interface_flat_Apr_10_2020.dat')
    mapping = readMapping('origin_data/pdb_pfam_mapping.txt')
    nodes_id, edges_id = set(), set()
    for node in nodes:
        if node in mapping.keys():
            nodes_id.add(mapping[node])
        else:
            print(node)
    for edge0, edge1, weight in edges:
        if edge0 in mapping.keys() and edge1 in mapping.keys():
            edges_id.add(tuple([mapping[edge0], mapping[edge1], weight]))
    with open('domain_graph', 'w') as f:
        for edge0, edge1, weight in edges_id:
            single_str = '\t'.join([edge0, edge1, str(weight)])+'\n'
            f.write(single_str)
