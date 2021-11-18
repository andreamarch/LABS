import json
import scipy.constants as consts
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import core.info as info
import core.utils as utils

root = Path(__file__).parent.parent
file = root / 'resources' / 'nodes.json'
out_dir = root / 'results'


#  STATIC ELEMENTS
class Node:
    def __init__(self, label, node_dict):
        self.label = label
        for el in node_dict:
            setattr(self, el, node_dict[el])
        self.successive = dict()

    def node_propagate(self, lightpath):
        flag = 0
        if len(lightpath.path) != 1:  # if not at the last node of the path
            # save the next line label
            line_label = lightpath.path[0] + lightpath.path[1]
            flag = 1
        # update the path deleting the current node
        lightpath.update_node(self)
        if flag == 1:
            # calling the next line propagate method
            self.successive[line_label].line_propagate(lightpath)
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
        self.state = [False] * 10  # false means 'free', 10 is the number of WDM channels

    def line_propagate(self, lightpath):
        # save the successive node label
        self.state[lightpath.channel] = True
        node_label = self.label[1]
        # generate the noise acquired on this line
        self.noise_generation(lightpath)
        # generate the latency acquired on this line
        self.latency_generation(lightpath)
        # call the successive node
        self.successive[node_label].node_propagate(lightpath)
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
        self.weighted_lines = utils.weighted_lines_build(self)  # build the weighted_lines data frame
        self.route_space = utils.route_space_build(self)  # build the route_space data frame
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
                    occupancy = []
                    channel = None
                    for index in range(0, len(path) - 1):  # checking if one of the lines in the path is occupied
                        label = path[index] + path[index + 1]
                        vect = [False] * len(self.lines[label].state)
                        for state_index in range(0, len(self.lines[label].state)):
                            if self.lines[label].state[state_index]:  # if it's true it's occupied
                                vect[state_index] = True  # set the flag vector true for the current index
                        occupancy.append(vect)
                    for j in range(0, len(occupancy[0])):  # cycle on the columns (channels)
                        flag2 = False
                        for i in range(0, len(occupancy)):  # cycle on the rows (lines of the path)
                            if occupancy[i][j]:  # channel occupied
                                flag2 = True
                        if not flag2:
                            # if the flag is still true after the cycle on the lines,
                            # it means that this channel is available for all the lines of the path, so it can be used
                            channel = j
                            flag = True
                            break
                    if flag:  # if all the channels of one line were occupied, the path and latency won't be saved
                        free_channel = channel
                        best_snr = float(row.SNR)
                        best_path = path
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
                    occupancy = []
                    channel = None
                    for index in range(0, len(path) - 1):  # checking if one of the lines in the path is occupied
                        label = path[index] + path[index + 1]
                        vect = [False] * len(self.lines[label].state)
                        for state_index in range(0, len(self.lines[label].state)):
                            if self.lines[label].state[state_index]:  # if it's true it's occupied
                                vect[state_index] = True
                        occupancy.append(vect)
                    for j in range(0, len(occupancy[0])):
                        flag2 = False
                        for i in range(0, len(occupancy)):
                            if occupancy[i][j]:  # channel occupied
                                flag2 = True
                        if not flag2:
                            channel = j
                            flag = True
                            break
                    if flag:  # if one of the lines was occupied, the path and latency won't be saved
                        best_latency = float(row.Latency)
                        best_path = path
                        free_channel = channel
        return best_latency, best_path, free_channel

    def propagate(self, light_path):
        first_node = light_path.path[0]
        self.nodes[first_node].node_propagate(light_path)  # calling the propagate method for the first node
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
        plt.savefig(out_dir / 'wgraph.png')
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
                if best_lat_path != '':  # if an available path was found, the string shouldn't be empty
                    signal = info.Lightpath(i.signal_power, best_lat_path, channel)  # initiate the Lightpath instance
                    signal = (self.propagate(signal))  # propagate the signal info through the path
                    utils.route_space_build(self)
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
                if best_snr_path != '':  # if an available path was found, the string shouldn't be empty
                    signal = info.Lightpath(i.signal_power, best_snr_path, channel)  # initiate the Lightpath instance
                    signal = (self.propagate(signal))  # propagate the signal info through the path
                    utils.route_space_build(self)
                    i.snr = 10 * np.log10(signal.signal_power / signal.noise_power)
                    i.latency = signal.latency
                else:  # no lines available between these nodes
                    i.snr = 0
                    i.latency = np.NaN
                    print('Connection blocked')
            else:  # wrong string was passed
                print('Please choose between strings "snr" and "latency".')
        for lbl in self.lines.keys():
            self.lines[lbl].state = [False] * 10  # at the end of the stream, the lines are again free
        return


class Connection:
    def __init__(self, inp, output, signal_power):
        self.input = inp.upper()
        self.output = output.upper()
        self.signal_power = signal_power
        self.snr = 0.0
        self.latency = 0.0
