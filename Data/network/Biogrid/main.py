def read_nodes_edges(path):
    with open(path, 'r')as f:
        nodes, edges = set(), set()
        for line in f:
            if ("(" or ")") in line:
                continue
            line_splited = line.strip().split('\t')
            edges.add(tuple(line_splited[:2]))
            nodes.add(line_splited[0])
            nodes.add(line_splited[1])
    return nodes, edges


def save_set(datas, path):
    with open(path, 'w')as f:
        for item in datas:
            single_line = (item if isinstance(item, str)
                           else ' '.join(item))+'\n'  # 必须用空格，由于后面的coach等方法调用
            f.write(single_line)


if __name__ == "__main__":
    nodes, edges = read_nodes_edges(
        "origin_data/biogrid_yeast_physical_unweighted.txt")
    save_set(nodes, "nodes")
    save_set(edges, 'edges')
