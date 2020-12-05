def readBioText(path, split, cut):
    res = []
    with open(path, 'r') as f:
        tempinfo = {}
        for index, line in enumerate(f):
            if index < 43 or index >= 6292:
                continue
            line = line.strip()
            if line == split:
                res.append(tempinfo)
                tempinfo = {}
            else:
                datas = line.split(cut)
                if len(datas) == 2:
                    key, val = datas
                    tempinfo[key] = tempinfo.get(key, [])
                    tempinfo[key].append(val)
    return res


if __name__ == "__main__":
    datas = readBioText('origin_data/subcell.txt', r'//', '   ')
    edges = list()
    mapping = dict()
    for item in datas:
        key = None
        key = item['ID'] if (key is None and 'ID' in item.keys()) else key
        key = item['IT'] if (key is None and 'IT' in item.keys()) else key
        key = item['IO'] if (key is None and 'IO' in item.keys()) else key
        key = key[0].replace('.', '')
        val = item['AC']
        if key:
            mapping[key] = val[0]
            if 'HI' in item.keys():
                for it in item['HI']:
                    target = it.replace('.', '')
                    edges.append([key, target, 'i'])
            if 'HP' in item.keys():
                for it in item['HP']:
                    target = it.replace('.', '')
                    edges.append([key, target, 'p'])
    with open("subcell_graph", 'w')as f:
        for v0, v1, typ in edges:
            singleline = '\t'.join([v0, v1, typ])+'\n'
            f.write(singleline)
