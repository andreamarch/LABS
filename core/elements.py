import json
import math
import random

import scipy.constants as consts
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import core.info as info
import core.utils as utils
import pandas as pd
from scipy.special import erfcinv
from core.parameters import *
from scipy.spatial import ConvexHull
from descartes import PolygonPatch

# import/export paths
root = Path(__file__).parent.parent
out_dir = root / 'results'


#  STATIC ELEMENTS
class Node:
    def __init__(self, label, node_dict):
        self.label = label
        self.transceiver = strategy_bit_rate
        for el in node_dict:
            setattr(self, el, node_dict[el])
        self.successive = dict()

    def node_propagate(self, lightpath, current_path):
        flag = 0
        current_node = lightpath.path[0]
        index = current_path.index(current_node)
        if (index != 0) & (index != (len(current_path) - 1)):
            previous_node = current_path[index - 1]
            next_node = current_path[index + 1]
            if lightpath.channel != (len(self.switching_matrix[previous_node][next_node]) - 1):
                self.switching_matrix[previous_node][next_node][lightpath.channel + 1] = 0
            if lightpath.channel != 0:
                self.switching_matrix[previous_node][next_node][lightpath.channel - 1] = 0
        if len(lightpath.path) != 1:  # if not at the last node of the path
            # save the next line label
            line_label = lightpath.path[0] + lightpath.path[1]
            flag = 1
            lightpath.signal_power = self.successive[line_label].optimized_launch_power()
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
            signalinformation.signal_power = self.successive[line_label].optimized_launch_power()
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
        self.state = np.ones(number_of_channels, dtype=int)  # 1 means 'free'
        self.n_amplifiers = int(np.ceil(length / span_length / 1e3) + 1)  # number of amplifiers, 1 amplifier every 80km
        self.n_span = self.n_amplifiers - 1  # number of fiber spans
        self.eta = 16 / (27 * math.pi) * np.log(
            (math.pi ** 2) / 2 * (beta2 * (sym_rate ** 2)) / alpha_m * (
                    10 ** (2 * sym_rate / df))) * alpha_m / beta2 * ((gamma ** 2) * ((l_eff * 1e3) ** 2)) / (
                           sym_rate ** 3)  # eta, formula from 08_NLI.pdf, slide 38

    def line_propagate(self, lightpath, current_path):
        # save the successive node label
        self.state[lightpath.channel] = 0
        node_label = self.label[1]
        # compute the average power on the current path
        lightpath.update_signal_pow_average()
        # generate the noise acquired on this line
        new_noise = self.noise_generation(lightpath)
        # compute new ISNR and update
        self.isnr_generation(lightpath, new_noise)
        # generate the latency acquired on this line
        self.latency_generation(lightpath)
        # call the successive node
        self.successive[node_label].node_propagate(lightpath, current_path)
        return lightpath

    def probe(self, signalinformation):
        # save the successive node label
        node_label = self.label[1]
        # compute the average power on the current path
        signalinformation.update_signal_pow_average()
        # generate the noise acquired on this line
        new_noise = self.noise_generation(signalinformation)
        # compute new ISNR and update
        self.isnr_generation(signalinformation, new_noise)
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
        new_ase = self.ase_generation()
        new_nli = self.nli_generation(signal_information, signal_information.signal_power)
        new_noise = new_nli + new_ase
        # modify noise in the signal information
        signal_information.update_noise_pow(new_noise)
        return new_noise

    def isnr_generation(self, signal_information, noise):
        new_isnr = noise / signal_information.signal_power
        signal_information.update_isnr(new_isnr)
        return

    def ase_generation(self):
        ase = self.n_amplifiers * (h * f0 * noise_bw * n_figure * (gain - 1))
        return ase

    def nli_generation(self, lightpath, power=1):
        if power == 1:  # if nothing was passed as power, set to the lightpath power
            power = lightpath.signal_power
        nli = self.n_span * self.eta * (power ** 3) * noise_bw
        return nli

    def optimized_launch_power(self):
        p_ase = self.ase_generation()
        tmp_arg = p_ase / (2 * self.eta * self.n_span * noise_bw)
        optimum_power = tmp_arg ** (1 / 3)
        return optimum_power


class Network:
    def __init__(self, network_file):
        self.weighted_lines = []
        self.route_space = []
        self.nodes = dict()
        self.lines = dict()
        self.graph = dict()
        self.network_file = network_file
        # open the json file
        with open(self.network_file) as json_file:
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
                    # saving the row of the df corresponding to the current path
                    current_path = self.route_space.loc[self.route_space['Path'] == row.Path]
                    # saving channel occupation information
                    occupancy = np.array(current_path['Availability'])[0]
                    # looking for a free channel
                    for i in occupancy:
                        cnt += 1
                        if i == 1:
                            flag = True
                            break
                    if flag:  # if one of the lines was occupied, the path and latency won't be saved
                        best_snr = float(row.SNR)
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
                    # saving channel occupation information
                    occupancy = np.array(current_path['Availability'])[0]
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
        light_path.path_length = len(light_path.path) - 1
        self.nodes[first_node].node_propagate(light_path,
                                              current_path)  # calling the propagate method for the first node
        return light_path

    def probe(self, signal_information):
        first_node = signal_information.path[0]
        signal_information.path_length = len(signal_information.path) - 1
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
            xi = float(j.position[0]) * 1e-3
            x.append(xi)
            yi = float(j.position[1]) * 1e-3
            y.append(yi)
        coord_array = np.transpose(np.array([x, y]))
        hull = ConvexHull(coord_array)  # Get the boundary coordinates
        coord_array = coord_array[hull.vertices]
        if input_file_flag == 'exam':  # add the outer intersection points
            intersection1 = utils.line_intersections('AE', 'DB', self)
            intersection2 = utils.line_intersections('FG', 'CA', self)
            coord_array = np.vstack((coord_array[0], intersection1, coord_array[1:3], intersection2, coord_array[3:5]))
        x_array = coord_array[:, 0]
        y_array = coord_array[:, 1]
        network_area = utils.polygon_area(x_array, y_array)  # compute area
        print('The area of the network is around', '{:.2f}'.format(network_area), 'km^2')

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        plt.axis([-4e2, 6.5e2, -6e2, 6e2])
        plt.title('Weighted graph of the Network')
        plt.xlabel('x, km')
        plt.ylabel('y, km')
        plt.grid()
        for i in label:
            indx = label.index(i)
            for j in self.nodes[i].connected_nodes:
                for k in label:
                    if k == j:
                        indx2 = label.index(k)
                        xx = [x[indx2], x[indx]]
                        yy = [y[indx2], y[indx]]
                        plt.plot(xx, yy, 'g', linewidth=0.5)  # lines
        for xx, yy, s in zip(x, y, label):
            circ = plt.Circle((xx, yy), radius=0.4e2)  # circles
            circ.set_facecolor('c')
            circ.set_edgecolor('k')
            ax.txt = plt.text(xx - 0.2e2, yy - 0.2e2, s, fontsize=13)  # labels
            ax.add_patch(circ)
        # Save as png
        if save_my_figure:
            plt.savefig(out_dir / 'EXAM_res' / 'Network_Topology.png')
        fig2, ax2 = plt.subplots(1)
        ax2.set_aspect('equal')
        plt.axis([-4e2, 6.5e2, -6e2, 6e2])
        plt.title('Boundary points of the network')
        plt.xlabel('x, km')
        plt.ylabel('y, km')
        plt.grid()
        ax2.scatter(x_array, y_array, color='red')
        x_array = np.append(x_array, x_array[0])
        y_array = np.append(y_array, y_array[0])
        plt.plot(x_array, y_array, 'g', linewidth=1)
        if save_my_figure:
            plt.savefig(out_dir / 'EXAM_res' / 'simplified_network_for_hull.png')
        # Show the image
        plt.show()

    def stream(self, connection_list, lat_or_snr='latency'):
        lat_or_snr = lat_or_snr.lower()
        for i in range(0, len(connection_list)):
            current_connection = connection_list[i]
            inp = current_connection.input
            outp = current_connection.output
            if lat_or_snr == 'latency':
                vec = list(self.find_best_latency(inp, outp))
                best_lat_path = vec[1]
                channel = vec[2]
                if best_lat_path == '':
                    current_connection.connection_status = True
                elif best_lat_path != '':  # if an available path was found, the string shouldn't be empty
                    first_node = best_lat_path[0]
                    current_path = list(best_lat_path)
                    signal = info.Lightpath(best_lat_path, channel)  # initiate the Lightpath instance
                    signal = (self.propagate(signal, current_path))  # propagate the signal info through the path
                    signal.path = current_path  # re-instantiate the signal path because it was deleted by propagate
                    current_connection.bit_rate = self.calculate_bit_rate(signal, self.nodes[first_node].transceiver)
                    if current_connection.bit_rate == 0.0:
                        current_connection.connection_status = True
                    self.route_space = utils.route_space_build(self)
                    current_connection.snr = 10 * np.log10(np.power(signal.isnr, -1))
                    current_connection.latency = signal.latency
                if current_connection.connection_status:  # no lines available between these nodes
                    current_connection.snr = 0
                    current_connection.latency = np.NaN
                    current_connection.channel = None
                    current_connection.bit_rate = np.NaN
            elif lat_or_snr == 'snr':
                vec = list(self.find_best_snr(inp, outp))
                best_snr_path = vec[1]
                channel = vec[2]
                if best_snr_path == '':
                    current_connection.connection_status = True
                if best_snr_path != '':  # if an available path was found, the string shouldn't be empty
                    first_node = best_snr_path[0]
                    current_path = list(best_snr_path)
                    signal = info.Lightpath(best_snr_path, channel)  # initiate the Lightpath instance
                    signal = (self.propagate(signal, current_path))  # propagate the signal info through the path
                    signal.path = current_path  # re-instantiate the signal path because it was deleted by propagate
                    current_connection.bit_rate = self.calculate_bit_rate(signal, self.nodes[first_node].transceiver)
                    if current_connection.bit_rate == 0.0:
                        current_connection.connection_status = True
                    self.route_space = utils.route_space_build(self)
                    current_connection.snr = 10 * np.log10(np.power(signal.isnr, -1))
                    current_connection.latency = signal.latency
                if current_connection.connection_status:  # no lines available between these nodes
                    current_connection.snr = 0
                    current_connection.latency = np.NaN
                    current_connection.channel = None
                    current_connection.bit_rate = np.NaN
            else:  # wrong string was passed
                print('Please choose between strings "snr" and "latency".')
        # print('There were', count_block_events, 'blocking events out of', number_of_connections, 'connections')
        return connection_list

    def calculate_bit_rate(self, lightpath, strategy):
        path_string = ''
        path = lightpath.path
        for k in range(0, len(path)):
            path_string += path[k]
            if k < len(path) - 1:
                path_string += '->'
        bit_rate = 0
        symb_rate = lightpath.sym_rate
        ber = 1e-3
        row_index = self.weighted_lines.index[self.weighted_lines['Path'] == path_string].tolist()
        gsnr = self.weighted_lines.loc[row_index, 'SNR'].tolist()[0]  # in dB
        gsnr = 10 ** (gsnr / 10)  # in linear scale
        if strategy == 'fixed_rate':
            condition = 2 * (erfcinv(2 * ber) ** 2) * (symb_rate / noise_bw)
            if gsnr >= condition:
                bit_rate = 100.0
            else:
                bit_rate = 0.0
        if strategy == 'flex_rate':
            condition0 = 2 * (erfcinv(2 * ber) ** 2) * (symb_rate / noise_bw)
            condition1 = 14 / 3 * (erfcinv(3 / 2 * ber) ** 2) * (symb_rate / noise_bw)
            condition2 = 10 * (erfcinv(8 / 3 * ber) ** 2) * (symb_rate / noise_bw)
            if gsnr < condition0:
                bit_rate = 0.0
            elif (gsnr >= condition0) & (gsnr < condition1):
                bit_rate = 100.0
            elif (gsnr >= condition1) & (gsnr < condition2):
                bit_rate = 200.0
            elif gsnr >= condition2:
                bit_rate = 400.0
        if strategy == 'shannon':
            bit_rate = 2 * symb_rate * np.log2(1 + gsnr * symb_rate / noise_bw) / 1e9
        return bit_rate

    def deploy_traffic_matrix(self, traffic_matrix, optimise_block_events=False, lat_or_snr='snr'):
        connections = []
        iteration_number = 0
        saturation_flag = False
        block_event_count = 0
        length_traffic_matrix = 0
        for i in traffic_matrix.keys():
            length_traffic_matrix += len(traffic_matrix[i])
        while len(traffic_matrix) > 0 and not saturation_flag:
            in_node = random.choice(list(traffic_matrix.keys()))
            out_node = random.choice(list(traffic_matrix[in_node].keys()))
            iteration_number += 1
            current_connection = Connection(in_node, out_node)
            current_connection = self.stream([current_connection], lat_or_snr)
            current_connection = current_connection[0]
            connections.append(current_connection)
            # Successful connection: subtract its bit rate from the request
            if not current_connection.connection_status:
                traffic_matrix[in_node][out_node] -= current_connection.bit_rate
                if traffic_matrix[in_node][out_node] <= 0:
                    traffic_matrix[in_node].pop(out_node)  # all traffic allocated
            # Connection blocked: remove the request from the traffic matrix
            elif current_connection.connection_status:
                block_event_count += 1
                if optimise_block_events:
                    # if the flag is true, the matrix element is removed to avoid other connection attempts between
                    # those nodes. False is default.
                    traffic_matrix[in_node].pop(out_node)
            if len(traffic_matrix[in_node]) == 0:
                traffic_matrix.pop(in_node)
            if block_event_count >= np.ceil(iteration_number * be_threshold):
                saturation_flag = True
        return connections, saturation_flag


class Connection:
    def __init__(self, inp, output, signal_power=None):
        self.input = inp.upper()
        self.output = output.upper()
        self.signal_power = signal_power if signal_power else 0
        self.snr = 0.0
        self.latency = 0.0
        self.bit_rate = 0.0
        self.connection_status = False  # False for successful connection, True for blocked connection
