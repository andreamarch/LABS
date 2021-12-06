import json
import scipy.constants as consts
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import core.info as info
import core.utils as utils
import pandas as pd
from scipy.special import erfcinv


# import/export paths
root = Path(__file__).parent.parent
file = root / 'resources' / 'nodes_not_full.json'
out_dir = root / 'results'


#  STATIC ELEMENTS
class Node:
    def __init__(self, label, node_dict):
        self.label = label
        self.transceiver = 'fixed_rate'
        for el in node_dict:
            setattr(self, el, node_dict[el])
        self.successive = dict()

    def node_propagate(self, lightpath, current_path):
        flag = 0
        current_node = lightpath.path[0]
        index = current_path.index(current_node)
        if (index != 0) & (index != (len(current_path) - 1)):
            previous_node = current_path[index-1]
            next_node = current_path[index+1]
            if lightpath.channel != (len(self.switching_matrix[previous_node][next_node]) - 1):
                self.switching_matrix[previous_node][next_node][lightpath.channel+1] = 0
            if lightpath.channel != 0:
                self.switching_matrix[previous_node][next_node][lightpath.channel-1] = 0
        if len(lightpath.path) != 1:  # if not at the last node of the path
            # save the next line label
            line_label = lightpath.path[0] + lightpath.path[1]
            flag = 1
        # update the path deleting the current node
        lightpath.update_node(self)
        if flag == 1:
            # calling the next line propagate method
            self.successive[line_label].line_propagate(lightpath, current_path)
        return lightpath

    def probe(self, signalinformation):
        flag = 0
        if len(signalinformation.path) != 1:  # if not at the last node of the path
            # save the next line label
            line_label = signalinformation.path[0] + signalinformation.path[1]
            flag = 1
        # update the path deleting the current node
        signalinformation.update_node(self)
        if flag == 1:
            # calling the next line probe method
            self.successive[line_label].probe(signalinformation)
        return signalinformation


class Line:
    def __init__(self, label, length):
        self.label = label
        self.length = length
        self.successive = dict()
        self.state = np.ones(10, dtype=int)  # 1 means 'free', 10 is the number of WDM channels

    def line_propagate(self, lightpath, current_path):
        # save the successive node label
        self.state[lightpath.channel] = 0
        node_label = self.label[1]
        # generate the noise acquired on this line
        self.noise_generation(lightpath)
        # generate the latency acquired on this line
        self.latency_generation(lightpath)
        # call the successive node
        self.successive[node_label].node_propagate(lightpath, current_path)
        return lightpath

    def probe(self, signalinformation):
        # save the successive node label
        node_label = self.label[1]
        # generate the noise acquired on this line
        self.noise_generation(signalinformation)
        # generate the latency acquired on this line
        self.latency_generation(signalinformation)
        # call the successive node probe
        self.successive[node_label].probe(signalinformation)
        return signalinformation

    def latency_generation(self, signal_information):
        # compute the latency
        speed = consts.c * 2 / 3
        new_latency = self.length / speed
        # modify the latency in the signal information
        signal_information.update_latency(new_latency)
        return signal_information

    def noise_generation(self, signal_information):
        # compute noise
        new_noise = 1e-9 * signal_information.signal_power * self.length
        # modify noise in the signal information
        signal_information.update_noise_pow(new_noise)
        return signal_information


class Network:
    def __init__(self):
        self.weighted_lines = []
        self.route_space = []
        self.nodes = dict()
        self.lines = dict()
        self.graph = dict()
        # open the json file
        with open(file) as json_file:
            nodes = json.load(json_file)
        # initiate the Node instances from the json file
        for i, j in nodes.items():
            self.nodes[i] = Node(i, j)
            temp_dict = dict()
            for x in self.nodes[i].connected_nodes:  # initiating a graph dictionary for the Dijkstra's algorithm
                temp_dict[x] = 1
            self.graph[i] = temp_dict

        for curr_node, i in nodes.items():  # cycling on the values of attribute nodes
            for next_node in i['connected_nodes']:  # cycling on the 'connected_nodes' key of the current value
                line_label = curr_node + next_node
                # save the positions of the current and next nodes
                x1 = self.nodes[curr_node].position[0]
                y1 = self.nodes[curr_node].position[1]
                x2 = self.nodes[next_node].position[0]
                y2 = self.nodes[next_node].position[1]
                # compute the length of the line
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # save the length as attribute of the instance of the line
                self.lines[line_label] = Line(line_label, line_length)

    def connect(self):
        for node_label, node_value in self.nodes.items():
            for line_label, line_value in self.lines.items():
                if line_label[0] == node_label:  # while cycling on the lines, if we have that the first node of the
                    # line corresponds to the current node, save the line in the successive of that node
                    node_value.successive[line_label] = line_value
                for i, j in self.nodes.items():
                    if i == line_label[1]:  # while cycling on the nodes, if the line corresponds to the second
                        # element of the line, save the node in the successive of the current line
                        line_value.successive[i] = j
        # build the weighted_lines data frame
        self.weighted_lines = utils.weighted_lines_build(self)
        # build the route_space data frame
        self.route_space = utils.route_space_build(self)
        return

    # implementing Dijkstra's algorithm in order to find all paths
    def find_paths(self, strt, end, path=[]):
        strt = strt.upper()
        end = end.upper()
        path = path + [strt]
        if strt == end:  # we reached the end
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
        best_snr = 0  # set to zero because it has to find the highest value of snr
        best_path = ''
        free_channel = None
        for row in self.weighted_lines.itertuples():
            if row.FromTo == from_to:  # using the FromTo column of the data structure weighted_lines
                if row.SNR > best_snr:
                    path = list(row.Path.replace('->', ''))
                    flag = False
                    cnt = -1
                    current_path = self.route_space.loc[self.route_space['Path'] == row.Path]
                    occupancy = current_path.loc[:, current_path.columns != 'Path']
                    occupancy = pd.DataFrame.to_numpy(occupancy, dtype=int)[0]
                    for i in occupancy:
                        cnt += 1
                        if i == 1:
                            flag = True
                            break
                    if flag:  # if one of the lines was occupied, the path and latency won't be saved
                        best_snr = float(row.snr)
                        best_path = path
                        free_channel = cnt
        return best_snr, best_path, free_channel

    def find_best_latency(self, strt, end):
        from_to = strt.upper() + '-' + end.upper()
        best_latency = 1e99  # set to infinite because it has to find the lowest value of latency
        best_path = ''
        free_channel = None
        for row in self.weighted_lines.itertuples():
            if row.FromTo == from_to:  # using the FromTo column of the data structure weighted_lines
                if row.Latency < best_latency:
                    path = list(row.Path.replace('->', ''))
                    flag = False
                    cnt = -1
                    # saving the row of the df corresponding to the current path
                    current_path = self.route_space.loc[self.route_space['Path'] == row.Path]
                    # saving the columns containing the info about the channel occupation
                    occupancy = current_path.loc[:, current_path.columns != 'Path']
                    occupancy = pd.DataFrame.to_numpy(occupancy, dtype=int)[0]
                    # looking for a free channel
                    for i in occupancy:
                        cnt += 1
                        if i == 1:
                            flag = True
                            break
                    if flag:  # if one of the lines was occupied, the path and latency won't be saved
                        best_latency = float(row.Latency)
                        best_path = path
                        free_channel = cnt
        return best_latency, best_path, free_channel

    def propagate(self, light_path, current_path):
        first_node = light_path.path[0]
        self.nodes[first_node].node_propagate(light_path, current_path)  # calling the propagate method for the first node
        return light_path

    def probe(self, signal_information):
        first_node = signal_information.path[0]
        self.nodes[first_node].probe(signal_information)  # calling the probe method for the first node
        return signal_information

    def draw_network(self):
        x = list()
        y = list()
        label = list()
        for i, j in self.nodes.items():
            # saving labels in vector
            label.append(i)
            # saving positions in vector
            xi = float(j.position[0]) * 1e-5
            x.append(xi)
            yi = float(j.position[1]) * 1e-5
            y.append(yi)
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        plt.axis([-5, 5, -5, 5.5])
        plt.title('Weighted graph of the Network')
        plt.xlabel('x, 100*km')
        plt.ylabel('y, 100*km')
        for i in label:
            indx = label.index(i)
            for j in self.nodes[i].connected_nodes:
                for k in label:
                    if k == j:
                        indx2 = label.index(k)
                        xx = [x[indx2], x[indx]]
                        yy = [y[indx2], y[indx]]
                        plt.plot(xx, yy, 'k-.', linewidth=0.5)  # lines
        for xx, yy, s in zip(x, y, label):
            circ = plt.Circle((xx, yy), radius=0.5)  # circles
            circ.set_facecolor('c')
            circ.set_edgecolor('k')
            ax.txt = plt.text(xx - 0.13, yy - 0.13, s, fontsize=14)  # labels
            ax.add_patch(circ)
        # Save as png
        plt.savefig(out_dir / '7_wgraph.png')
        # Show the image
        plt.show()

    def stream(self, connection_list, lat_or_snr='latency'):
        lat_or_snr = lat_or_snr.lower()
        for i in connection_list:
            inp = i.input
            outp = i.output
            if lat_or_snr == 'latency':
                vec = list(self.find_best_latency(inp, outp))
                best_lat_path = vec[1]
                channel = vec[2]
                if best_lat_path != '':
                    first_node = best_lat_path[0]
                    i.bit_rate = self.calculate_bit_rate(best_lat_path, self.nodes[first_node].transceiver)
                    if i.bit_rate == 0.0:
                        best_lat_path = ''
                if best_lat_path != '':  # if an available path was found, the string shouldn't be empty
                    current_path = list(best_lat_path)
                    signal = info.Lightpath(i.signal_power, best_lat_path, channel)  # initiate the Lightpath instance
                    signal = (self.propagate(signal, current_path))  # propagate the signal info through the path
                    utils.route_space_update(self, current_path)
                    i.snr = 10 * np.log10(signal.signal_power / signal.noise_power)
                    i.latency = signal.latency
                else:  # no lines available between these nodes
                    i.snr = 0
                    i.latency = np.NaN
                    print('Connection blocked')
            elif lat_or_snr == 'snr':
                vec = list(self.find_best_snr(inp, outp))
                best_snr_path = vec[1]
                channel = vec[2]
                if best_snr_path != '':
                    first_node = best_snr_path[0]
                    i.bit_rate = self.calculate_bit_rate(best_snr_path, self.nodes[first_node].transceiver)
                    if i.bit_rate == 0.0:
                        best_snr_path = ''
                if best_snr_path != '':  # if an available path was found, the string shouldn't be empty
                    current_path = list(best_snr_path)
                    signal = info.Lightpath(i.signal_power, best_snr_path, channel)  # initiate the Lightpath instance
                    signal = (self.propagate(signal, current_path))  # propagate the signal info through the path
                    utils.route_space_update(self, current_path)
                    i.snr = 10 * np.log10(signal.signal_power / signal.noise_power)
                    i.latency = signal.latency
                else:  # no lines available between these nodes
                    i.snr = 0
                    i.latency = np.NaN
                    print('Connection blocked')
            else:  # wrong string was passed
                print('Please choose between strings "snr" and "latency".')
        for lbl in self.lines.keys():
            self.lines[lbl].state = [1] * 10  # at the end of the stream, the lines are again free
        with open(file) as json_file:
            nodes_from_file = json.load(json_file)
        # rewrite the switching matrix
        for i, j in nodes_from_file.items():
            for k in j:
                if k == 'switching_matrix':
                    self.nodes[i].switching_matrix = nodes_from_file[i][k]
        return

    def calculate_bit_rate(self, path, strategy):
        path_string = ''
        for k in range(0, len(path)):
            path_string += path[k]
            if k < len(path) - 1:
                path_string += '->'
        bit_rate = 0
        sym_rate = 32e9
        ber = 1e-3
        noise_bw = 12.5e9
        row_index = self.weighted_lines.index[self.weighted_lines['Path'] == path_string].tolist()
        gsnr = self.weighted_lines.loc[row_index, 'SNR'].tolist()[0]
        if strategy == 'fixed_rate':
            condition = 2 * (erfcinv(2*ber) ** 2) * (sym_rate / noise_bw)
            if gsnr >= condition:
                bit_rate = 100.0
            else:
                bit_rate = 0.0
        if strategy == 'flex_rate':
            condition0 = 2 * (erfcinv(2 * ber) ** 2) * (sym_rate / noise_bw)
            condition1 = 14 / 3 * (erfcinv(3 / 2 * ber) ** 2) * (sym_rate / noise_bw)
            condition2 = 10 * (erfcinv(8 / 3 * ber) ** 2) * (sym_rate / noise_bw)
            if gsnr < condition0:
                bit_rate = 0.0
            elif (gsnr >= condition0) & (gsnr < condition1):
                bit_rate = 100.0
            elif (gsnr >= condition1) & (gsnr < condition2):
                bit_rate = 200.0
            elif gsnr >= condition2:
                bit_rate = 400.0
        if strategy == 'shannon':
            bit_rate = 2 * sym_rate * np.log2(1 + gsnr * sym_rate / noise_bw) / 1e9
        return bit_rate


class Connection:
    def __init__(self, inp, output, signal_power):
        self.input = inp.upper()
        self.output = output.upper()
        self.signal_power = signal_power
        self.snr = 0.0
        self.latency = 0.0
        self.bit_rate = 0.0
