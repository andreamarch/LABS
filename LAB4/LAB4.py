import json
import scipy.constants as consts
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class SignalInformation:
    def __init__(self, signal_power, path, noise_power=0.0, latency=0.0):
        self.signal_power = float(signal_power)
        self.path = path
        self.noise_power = noise_power
        self.latency = latency

    def update_signal_pow(self, signal_power_increment):
        self.signal_power += signal_power_increment
        return

    def update_noise_pow(self, noise_power_increment):
        self.noise_power += noise_power_increment
        return

    def update_latency(self, latency_increment):
        self.latency += latency_increment
        return

    def update_node(self, node):
        if self.path[0] == node.label:
            self.path.remove(self.path[0])
        return self.path


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


class Line:
    def __init__(self, label, length):
        self.label = label
        self.length = length
        self.successive = dict()
        self.status = 'occupied'

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


class Network:
    def __init__(self):
        self.weighted_lines = []
        self.nodes = dict()
        self.lines = dict()
        self.graph = dict()
        with open('nodes.json') as json_file:
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
        self.weighted_lines = weigthed_nodes_build(self)
        return

    # implementing Dijkstra's algorithm in order to find all paths
    def find_paths(self, strt, end, path=[]):
        strt = strt.upper()
        end = end.upper()
        path = path + [strt]
        if strt == end:
            return [path]
        if strt not in self.graph:
            return []
        paths = []
        for node in self.graph[strt]:
            if node not in path:
                newpaths = self.find_paths(node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def find_best_snr(self, strt, end):
        from_to = strt.upper() + '-' + end.upper()
        best_snr = 0
        best_path = ''
        for row in self.weighted_lines.itertuples():
            if row.FromTo == from_to:
                if row.SNR > best_snr:
                    path = list(row.Path.replace('->', ''))
                    flag = True
                    for index in range(0, len(path)-1):  # checking if one of the lines in the path is occupied
                        label = path[index] + path[index+1]
                        if self.lines[label].status == 'occupied':
                            flag = False
                    if flag:  # if one of the lines was occupied, the path and latency won't be saved
                        best_snr = float(row.SNR)
                        best_path = path
        return best_snr, best_path

    def find_best_latency(self, strt, end):
        from_to = strt.upper() + '-' + end.upper()
        best_latency = 1e99
        best_path = ''
        for row in self.weighted_lines.itertuples():
            if row.FromTo == from_to:
                if row.Latency < best_latency:
                    path = list(row.Path.replace('->', ''))
                    flag = True
                    for index in range(0, len(path) - 1):  # checking if one of the lines in the path is occupied
                        label = path[index] + path[index + 1]
                        if self.lines[label].status == 'free':
                            flag = False
                    if flag:  # if one of the lines was occupied, the path and latency won't be saved
                        best_latency = float(row.Latency)
                        best_path = path
        return best_latency, best_path

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

    def stream(self, connection_list, lat_or_snr='latency'):
        lat_or_snr = lat_or_snr.lower()
        for i in connection_list:
            inp = i.input
            outp = i.output
            if lat_or_snr == 'latency':
                best_lat_path = self.find_best_latency(inp, outp)[1]
                if best_lat_path != '':  # if an available path was found, the string shouldn't be empty
                    signal = SignalInformation(i.signal_power, best_lat_path)
                    signal = (self.propagate(signal))
                    i.snr = 10 * np.log10(signal.signal_power / signal.noise_power)
                    i.latency = signal.latency
                else:
                    i.snr = 0
                    i.latency = None
            elif lat_or_snr == 'snr':
                best_snr_path = self.find_best_snr(inp, outp)[1]
                if best_snr_path != '':
                    signal = SignalInformation(i.signal_power, best_snr_path)
                    signal = (self.propagate(signal))
                    i.snr = 10 * np.log10(signal.signal_power / signal.noise_power)
                    i.latency = signal.latency
                else:
                    i.snr = 0
                    i.latency = None
            else:
                print('Please choose between strings "snr" and "latency".')
        return

# function for constructing the data frame
def weigthed_nodes_build(network, power = 1e-3):
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
    df = pd.DataFrame(data, columns=['FromTo', 'Path', 'Power', 'NoisePower', 'SNR', 'Latency'])
    return df

class Connection:
    def __init__(self, inp, output, signal_power):
        self.input = inp.upper()
        self.output = output.upper()
        self.signal_power = signal_power
        self.snr = 0.0
        self.latency = 0.0


network = Network()
network.connect()
nodes = list(network.nodes.keys())
connections = []
signal_power = 1e-3
for i in range(0, 99):
    new_nodes = list(nodes)
    strt = random.choice(new_nodes)
    new_nodes.remove(strt)
    end = random.choice(new_nodes)
    connections.append(Connection(strt, end, signal_power))
network.stream(connections, 'latency')
snr = []
lat = []
for i in range(0, 99):
    snr.append(connections[i].snr)
    lat.append(connections[i].latency)
plt.figure(1)
plt.plot(snr)
plt.figure(2)
plt.plot(lat)
plt.show()
# connection = [Connection('a', 'e', 1e-3), Connection('c', 'f', 1e-3)]
# network.stream(connection, 'latency')



