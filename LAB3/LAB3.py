import json
import scipy.constants as consts
from pathlib import Path

# ex 1
class SignalInformation:

    def __init__(self, signal_power, path, noise_power=0.0, latency=0.0):
        self.signal_power = float(signal_power)
        self.path = list(path.upper())
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
        latency_status = True
        return latency_status

    def update_node(self):
        self.path.remove(self.path[0])
        return


# ex 2
class Node:
    def __init__(self, node_dict, label):
        self.label = label
        for el in node_dict:
            setattr(self, el, node_dict[el])
        self.successive = dict()

    def propagation(self):
        SignalInformation.update_node(SignalInformation)
        return

# ex 3
class Line:
    def __init__(self, label, length):
        self.label = label
        self.length = length
        self.successive = dict()

    def latency_generation(self):
        speed = consts.c * 2/3
        new_latency =  self.length / speed
        SignalInformation.update_latency(SignalInformation, new_latency)
        return

signal = SignalInformation(9,'abcd')
line = Line('AB', 1)
line.latency_generation()