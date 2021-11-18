import numpy as np
import pandas as pd
import core.info as info


# function for constructing the data frame
def weighted_lines_build(network, power=1e-3):
    from_to = []
    paths = []
    signal_pow = []
    noise = []
    latency = []
    path_vector = []
    snr = []
    for i in network.nodes.keys():  # building vector of paths
        for j in network.nodes.keys():
            if j != i:
                paths.append(network.find_paths(i, j))

    for i in range(0, len(paths)):
        for j in range(0, len(paths[i])):
            strng = ''  #  building the string in the format with ->
            for k in range(0, len(paths[i][j])):
                strng += paths[i][j][k]
                if k < len(paths[i][j]) - 1:
                    strng += '->'
            path_vector.append(strng)
            from_to.append(paths[i][j][0] + '-' + paths[i][j][len(paths[i][j]) - 1])
            current_path = list(paths[i][j])
            signal_info = info.SignalInformation(power, current_path)
            network.probe(signal_info)
            signal_pow.append(signal_info.signal_power)
            noise.append(signal_info.noise_power)
            latency.append(signal_info.latency)
            snr.append(10 * np.log10(signal_info.signal_power / signal_info.noise_power))

    data = list(zip(from_to, path_vector, signal_pow, noise, snr, latency))
    df = pd.DataFrame(data, columns=['FromTo', 'Path', 'Power', 'NoisePower', 'SNR', 'Latency'])
    return df


# function for building and updating the route space dataframe
def route_space_build(network):
    labels = []
    occupancy = []
    for i, j in network.lines.items():
        labels.append(i)
        occupancy.append(j.state)
    df = pd.DataFrame(occupancy)
    df.insert(0, 'Line', labels, True)
    return df

