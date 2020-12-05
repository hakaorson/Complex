import time

import requests


def anasys_dip_item(item):
    splited = item.split('|')
    result = {'dip': None, 'refseq': None,  'uniprotkb': None}
    result['dip'] = splited[0]
    for item in splited[1:]:
        name, data = item.split(':')
        if name == 'refseq':
            result['refseq'] = data
        if name == 'uniprotkb':
            result['uniprotkb'] = data
    return result


def get_info(path):
    dipids, refids = set(), set()
    with open(path, 'r') as f:
        next(f)
        for line in f:
            list_info = line.strip().split('\t')
            l_info = anasys_dip_item(list_info[0])
            r_info = anasys_dip_item(list_info[1])
            if l_info["dip"]:
                dipids.add(l_info["dip"])
            if l_info["refseq"]:
                refids.add(l_info["refseq"])
            if r_info["dip"]:
                dipids.add(r_info["dip"])
            if r_info["refseq"]:
                refids.add(r_info["refseq"])
    return dipids, refids


def read_dip(path, dipmap, refmap):
    resPairs, resIds = set(), set()
    with open(path, 'r') as f:
        next(f)
        totalnomatch, nomatchnum, selfloopnum, repeatnum = 0, 0, 0, 0
        for line in f:
            list_info = line.strip().split('\t')
            l_info = anasys_dip_item(list_info[0])
            r_info = anasys_dip_item(list_info[1])
            l_id = l_info['uniprotkb']
            r_id = r_info['uniprotkb']
            if l_id is None or r_id is None:
                totalnomatch += 1
                if l_id is None:
                    l_id_dipmap = dipmap[l_info['dip']
                                         ] if l_info['dip'] in dipmap.keys() else None
                    l_id_refmap = refmap[l_info['refseq']
                                         ] if l_info['refseq'] in refmap.keys() else None
                    l_id = l_id_dipmap if l_id_dipmap is not None else l_id
                    l_id = l_id_refmap if l_id_refmap is not None else l_id
                if r_id is None:
                    r_id_dipmap = dipmap[r_info['dip']
                                         ] if r_info['dip'] in dipmap.keys() else None
                    r_id_refmap = refmap[r_info['refseq']
                                         ] if r_info['refseq'] in refmap.keys() else None
                    r_id = r_id_dipmap if r_id_dipmap is not None else r_id
                    r_id = r_id_refmap if r_id_refmap is not None else r_id

                if l_id is None or r_id is None:
                    print("No match:", l_id, r_id)
                    nomatchnum += 1
                    continue
            if l_id == r_id:
                # print("Loop:", l_id, r_id)
                selfloopnum += 1
                # continue loop应该记录下来
            if (l_id, r_id) in resPairs or (r_id, l_id) in resPairs:
                # print("Reapeat:", l_id, r_id)
                repeatnum += 1
                continue  # 有重复的
            resPairs.add((l_id, r_id))
            resIds.add(l_id)
            resIds.add(r_id)
        print("Total Not match:{},Not match:{},Loop:{},Reapeat:{},Lefted:{}".format(
            totalnomatch, nomatchnum, selfloopnum, repeatnum, len(resPairs)))
    return resPairs, resIds


def read_id(path):
    nodes = set()
    with open(path) as f:
        for line in f:
            linelist = line.strip().split('\t')
            for singleid in linelist:
                nodes.add(singleid)
    return nodes


def readmap(path):
    res = {}
    with open(path) as f:
        next(f)
        for line in f:
            linelist = line.strip().split('\t')
            keys = linelist[0].split(',')
            value = linelist[1]
            for key in keys:
                res[key] = value
    return res


def save_set(datas, path):
    with open(path, 'w')as f:
        for item in datas:
            single_line = (item if isinstance(item, str)
                           else ' '.join(item))+'\n'  # 必须用空格，由于后面的coach等方法调用
            f.write(single_line)


if __name__ == "__main__":
    source_path = 'origin_data/dip_22977.txt'
    dipids, refids = get_info(source_path)
    # uniprot处理
    for singleid in list(dipids):
        print(singleid, end=" ")
    print()
    dipmap = readmap("origin_data/dip_map")
    # uniprot处理，注意只选择对应生物的
    for singleid in list(refids):
        print(singleid, end=" ")
    print()
    # 注意通过控制colume选择想要的数据
    # https://www.uniprot.org/uniprot/?query=yourlist:M202012035C475328CEF75220C360D524E9D456CE06E477J&sort=yourlist:M202012035C475328CEF75220C360D524E9D456CE06E477J&columns=yourlist(M202012035C475328CEF75220C360D524E9D456CE06E477J),id
    refmap = readmap("origin_data/ref_map")
    # 需要先准备dipid的mapping和refid的mapping
    resPairs, resIds = read_dip(source_path, dipmap, refmap)
    save_set(resPairs, "edges")
    save_set(resIds, "nodes")
