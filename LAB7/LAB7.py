import core.elements as el
import random
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).parent.parent
in_directory = root / 'resources'
out_directory = root / 'results'

network = el.Network()
network.connect()
nodes = list(network.nodes.keys())
connections = []
signal_power = 1e-3
for i in range(0, 100):
    new_nodes = list(nodes)
    strt = random.choice(new_nodes)
    new_nodes.remove(strt)
    end = random.choice(new_nodes)
    connections.append(el.Connection(strt, end, signal_power))
network.stream(connections, 'latency')
snr = []
lat = []
br = []
for i in range(0, 99):
    snr.append(connections[i].snr)
    lat.append(connections[i].latency)
    br.append(connections[i].bit_rate)

plt.style.use('ggplot')

plt.figure(1)
plt.title('SNR distribution')
plt.hist(snr, bins=12)
plt.xlabel('SNR, dB')
plt.ylabel('Number of occurrences')

plt.figure(2)
plt.title('Latency distribution')
plt.hist(lat, bins=10)
plt.xlabel('Latency, s')
plt.ylabel('Number of occurrences')

plt.figure(3)
plt.title('Bit rate distribution')
plt.hist(br, bins=10)
plt.xlabel('Bit rate, Gbps')
plt.ylabel('Number of occurrences')

plt.show()


