import json
import scipy.constants as consts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

root = Path(__file__).parent.parent.parent
file = root / 'resources' / 'nodes.json'

# ex 1
class SignalInformation:

    def __init__(self, signal_power, path, noise_power=0.0, latency=0.0):
        self.signal_power = float(signal_power)
        self.path = path
        self.noise_power = noise_power
        self.latency = latency

    def update_signal_pow(self, signal_power_increment):
        self.signal_power += signal_power_increment
        return self.signal_power

    def update_noise_pow(self, noise_power_increment):
        self.noise_power += noise_power_increment
        return self.noise_power

    def update_latency(self, latency_increment):
        self.latency += latency_increment
        return self.lantency

    def update_node(self, node):
        if self.path[0] == node.label:
            self.path.remove(self.path[0])
        return self.path


# ex 2
class Node:
    def __init__(self, label, node_dict):
        self.label = label
        for el in node_dict:
            setattr(self, el, node_dict[el])
        self.successive = dict()

    def node_propagate(self, signalinformation):
        flag = 0
        if len(signalinformation.path) != 1:
            line_label = signalinformation.path[0] + signalinformation.path[1]
            flag = 1
        signalinformation.update_node(self)
        if flag == 1:
            self.successive[line_label].line_propagate(signalinformation)
        return signalinformation


# ex 3
class Line:
    def __init__(self, label, length):
        self.label = label
        self.length = length
        self.successive = dict()

    def line_propagate(self, signalinformation):
        node_label = self.label[1]
        self.noise_generation(signalinformation)
        self.latency_generation(signalinformation)
        self.successive[node_label].node_propagate(signalinformation)
        return signalinformation

    def latency_generation(self, signalinformation):
        speed = consts.c * 2 / 3
        new_latency = self.length / speed
        signalinformation.update_latency(new_latency)
        return signalinformation

    def noise_generation(self, signalinformation):
        new_noise = 1e-9 * signalinformation.signal_power * self.length
        signalinformation.update_noise_pow(new_noise)
        return signalinformation


# ex 4
class Network:
    def __init__(self):
        self.nodes = dict()
        self.lines = dict()
        self.graph = dict()
        with open(file) as json_file:
            nodes = json.load(json_file)
        for i, j in nodes.items():
            self.nodes[i] = Node(i, j)
            temp_dict = dict()
            for x in self.nodes[i].connected_nodes:
                temp_dict[x] = 1
            self.graph[i] = temp_dict

        for curr_node, i in nodes.items():
            for next_node in i['connected_nodes']:
                line_label = curr_node + next_node
                x1 = self.nodes[curr_node].position[0]
                y1 = self.nodes[curr_node].position[1]
                x2 = self.nodes[next_node].position[0]
                y2 = self.nodes[next_node].position[1]
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                self.lines[line_label] = Line(line_label, line_length)

    def connect(self):
        for node_label, node_value in self.nodes.items():
            for line_label, line_value in self.lines.items():
                if line_label[0] == node_label:
                    node_value.successive[line_label] = line_value
                for i, j in self.nodes.items():
                    if i == line_label[1]:
                        line_value.successive[i] = j
        return

    # implementing Dijkstra's algorithm in order to find all paths
    def find_paths(self, start, end, path=[]):
        start = start.upper()
        end = end.upper()
        path = path + [start]
        if start == end:
            return [path]
        if start not in self.graph:
            return []
        paths = []
        for node in self.graph[start]:
            if node not in path:
                newpaths = self.find_paths(node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def propagate(self, signalinformation):
        first_node = signalinformation.path[0]
        self.nodes[first_node].node_propagate(signalinformation)
        return signalinformation

    def draw_network(self):
        x = list()
        y = list()
        label = list()
        cnt = 0
        for i, j in self.nodes.items():
            label.append(i)
            xi = float(j.position[0]) * 1e-5
            x.append(xi)
            yi = float(j.position[1]) * 1e-5
            y.append(yi)

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        plt.axis([-5, 5, -5, 5.5])
        for i in label:
            indx = label.index(i)
            for j in self.nodes[i].connected_nodes:
                for k in label:
                    if k == j:
                        indx2 = label.index(k)
                        xx = [x[indx2], x[indx]]
                        yy = [y[indx2], y[indx]]
                        plt.plot(xx, yy, 'k-.', linewidth=0.5)
        for xx, yy, s in zip(x, y, label):
            circ = plt.Circle((xx, yy), radius=0.5)
            circ.set_facecolor('c')
            circ.set_edgecolor('k')
            ax.txt = plt.text(xx - 0.13, yy - 0.13, s, fontsize=14)
            ax.add_patch(circ)
        # Show the image
        plt.show()

# exercise 5
network = Network()
network.connect()
# network.draw_network()
power = 1e-3
from_to = []
paths = []
signal_pow = []
noise = []
latency = []
path_vector = []
snr = []
for i in network.nodes.keys():
    for j in network.nodes.keys():
        if j != i:
            paths.append(network.find_paths(i, j))

for i in range(0, len(paths)):
    for j in range(0, len(paths[i])):
        strng = ''
        for k in range(0, len(paths[i][j])):
            strng += paths[i][j][k]
            if k < len(paths[i][j]) - 1:
                strng += '->'
        path_vector.append(strng)
        from_to.append(paths[i][j][0] + '-' + paths[i][j][len(paths[i][j]) - 1])
        current_path = list(paths[i][j])
        signal_info = SignalInformation(power, current_path)
        network.propagate(signal_info)
        signal_pow.append(signal_info.signal_power)
        noise.append(signal_info.noise_power)
        latency.append(signal_info.latency)
        snr.append(10 * np.log10(signal_info.signal_power / signal_info.noise_power))

data = list(zip(from_to, path_vector, signal_pow, noise, snr, latency))
df = pd.DataFrame(data, columns=['From-To', 'Path', 'Power', 'Noise Power', 'SNR', 'Latency'])
print(df)
