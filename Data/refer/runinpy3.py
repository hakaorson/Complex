import os
import subprocess


class baseMethod():
    def __init__(self, method_name, graph_path, res_path, expand):
        self.method_path = (
            method_name if not expand else method_name+'_expand')+".py"
        self.graph_path = graph_path
        self.res_path = res_path

    def main(self, recompute=True):
        if recompute or not os.path.exists(self.res_path):
            cmd_res = self.run()
            self.getcomplexes(cmd_res)
        return data.read_bench(self.res_path)

    def run(self):
        assert(os.path.exists(self.method_path))
        cmd = "{} {} {}".format("python2.7", self.method_path, self.graph_path)
        # print(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res = []
        while p.poll() is None:
            line = p.stdout.readline().decode("utf8")
            res.append(line)
        return res

    def getcomplexes(self, cmd_res):
        ImportError


class ipca_method(baseMethod):
    def __init__(self, graph_path, res_path, expand):
        super().__init__("ipca", graph_path, res_path, expand)

    def getcomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for index, item in enumerate(cmd_res):
                if index % 2 == 0:
                    f.write(item)
                    # print(item)


class dpclus_method(baseMethod):
    def __init__(self, graph_path, res_path, expand):
        super().__init__("dpclus", graph_path, res_path, expand)

    def getcomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for index, item in enumerate(cmd_res):
                if index % 2 == 0:
                    f.write(item)


class clique_method(baseMethod):
    def __init__(self, graph_path, res_path, expand):
        super().__init__("clique", graph_path, res_path, expand)

    def getcomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for item in cmd_res[1:]:
                f.write(item)


class coach_method(baseMethod):
    def __init__(self, graph_path, res_path, expand):
        super().__init__("coach", graph_path, res_path, expand)

    def getcomplexes(self, cmd_res):
        with open(self.res_path, 'w') as f:
            for index, item in enumerate(cmd_res):
                f.write(item)


class mcode_method(baseMethod):
    def __init__(self, graph_path, res_path, expand):
        super().__init__("mcode", graph_path, res_path, expand)

    def getcomplexes(self, cmd_res):
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
                f.write(item)


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


def main(method_name, edges_path,  result_path, expand):
    methodor = get_method(method_name)
    meth = methodor(edges_path, result_path, expand)
    meth.main()


if __name__ == "__main__":
    pass
