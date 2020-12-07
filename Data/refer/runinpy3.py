import os
import subprocess


class baseMethod():
    def __init__(self, method_name, graph_path, res_path, expand, basedir):
        self.method_path = basedir+"/"+(
            method_name if not expand else method_name+'_expand')+".py"
        self.graph_path = graph_path
        self.res_path = res_path

    def run(self):
        print("computing refer_data")
        assert(os.path.exists(self.method_path))
        py2 = "D:\software\Anaconda\envs\python2.7\python.exe" if os.path.exists(
            "D:\software\Anaconda\envs\python2.7\python.exe") else "python2.7"
        cmd = [py2, self.method_path, self.graph_path]
        # print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res = []
        while p.poll() is None:
            line = p.stdout.readline().decode("utf8").strip()
            if len(line.strip()):  # 空行需要删除
                res.append(line)
        self.writecomplexes(res)
        return

    def writecomplexes(self, cmd_res):
        ImportError


class ipca_method(baseMethod):
    def __init__(self, graph_path, res_path, expand, basedir):
        super().__init__("ipca", graph_path, res_path, expand, basedir)

    def writecomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for index, item in enumerate(cmd_res):
                if index % 2 == 0:
                    f.write(item+'\n')
                    # print(item)


class dpclus_method(baseMethod):
    def __init__(self, graph_path, res_path, expand, basedir):
        super().__init__("dpclus", graph_path, res_path, expand, basedir)

    def writecomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for index, item in enumerate(cmd_res):
                if index % 2 == 0:
                    f.write(item+'\n')


class clique_method(baseMethod):
    def __init__(self, graph_path, res_path, expand, basedir):
        super().__init__("clique", graph_path, res_path, expand, basedir)

    def writecomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for item in cmd_res[1:]:
                f.write(item+'\n')


class coach_method(baseMethod):
    def __init__(self, graph_path, res_path, expand, basedir):
        super().__init__("coach", graph_path, res_path, expand, basedir)

    def writecomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for index, item in enumerate(cmd_res):
                f.write(item+'\n')


class mcode_method(baseMethod):
    def __init__(self, graph_path, res_path, expand, basedir):
        super().__init__("mcode", graph_path, res_path, expand, basedir)

    def writecomplexes(self, cmd_res):
        res = []
        start = False
        for item in cmd_res:
            if start:
                if len(item) and "\r" not in item:
                    res.append(item)
            if item == "molecular complex prediction\r\n":
                start = True
        with open(self.res_path, 'w') as f:
            for item in res:
                f.write(item+'\n')


def get_method(name):
    if name == "ipca":
        return ipca_method
    if name == "dpclus":
        return dpclus_method
    if name == "clique":
        return clique_method
    if name == "mcode":
        return mcode_method
    if name == "coach":
        return coach_method
    return None


def main(method_name, edges_path,  result_path, expand, basedir=""):
    methodor = get_method(method_name)
    meth = methodor(edges_path, result_path, expand, basedir)
    meth.run()


if __name__ == "__main__":
    pass
