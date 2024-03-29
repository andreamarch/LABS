import random
import core.elements as el
import matplotlib.pyplot as plt
from pathlib import Path
import core.utils as utls
from core.parameters import *

# import/export paths
root = Path(__file__).parent.parent.parent
if input_file_flag == 'lab':
    in_directory = root / 'resources'
    out_directory = root / 'results' / 'LAB10_res'
    file_fixed = in_directory / 'nodes_full_fixed_rate.json'
    file_flex = in_directory / 'nodes_full_flex_rate.json'
    file_shannon = in_directory / 'nodes_full_shannon.json'
elif input_file_flag == 'exam':
    in_directory = root / 'resources' / 'Exam_files'
    out_directory = root / 'results' / 'EXAM_res'
    file_fixed = in_directory / 'full_network_fixed.json'
    file_flex = in_directory / 'full_network_flex.json'
    file_shannon = in_directory / 'full_network_shannon.json'
else:
    raise ValueError('It can only be exam or lab...')

# network generation
network_fixed = el.Network(file_fixed)
network_fixed.connect()
network_flex = el.Network(file_flex)
network_flex.connect()
network_shan = el.Network(file_shannon)
network_shan.connect()

# network_fixed.draw_network()
node_degree = utls.compute_node_degree(network_fixed)
print('The average node degree for this network is', node_degree)
[avg_length, max_length, min_length] = utls.length_line_analysis(network_fixed)
print('The average line length for this network is ' + str(avg_length) + ' km')
print('The maximum line length is ' + str(max_length[0]) + ' km, corresponding to line ' + max_length[1])
print('The minimum line length is ' + str(min_length[0]) + ' km, corresponding to line ' + min_length[1])

# ----------------- POINT 1 (Single Traffic Matrix) -----------------
if do_simulations[0]:
    # output file paths
    outp_plot_avg_snr = out_directory / 'Point1' / '10d1_average_snr'
    outp_plot_min_snr = out_directory / 'Point1' / '10d1_min_snr'
    outp_plot_max_snr = out_directory / 'Point1' / '10d1_max_snr'
    outp_plot_capacity = out_directory / 'Point1' / '10d1_capacity'
    outp_plot_avg_br = out_directory / 'Point1' / '10d1_average_bit_rate'
    outp_plot_min_br = out_directory / 'Point1' / '10d1_min_bit_rate'
    outp_plot_max_br = out_directory / 'Point1' / '10d1_max_bit_rate'
    outp_plot_block_events = out_directory / 'Point1' / '10d1_blocking_events'
    outp_plot_connections = out_directory / 'Point1' / '10d1_successful_connections'
    outp_plot_congestion = out_directory / 'Point1' / '10d1_congestion'

    connections_fixed_per_M = []
    number_connections_fixed_rate_per_M = []
    number_blocking_events_fixed_rate_per_M = []
    capacities_fixed_rate_per_M = []
    average_bit_rate_fixed_rate_per_M = []
    minimum_snr_fixed_per_M = []
    maximum_snr_fixed_per_M = []
    average_snr_fixed_per_M = []

    connections_flex_per_M = []
    number_connections_flex_rate_per_M = []
    number_blocking_events_flex_rate_per_M = []
    capacities_flex_rate_per_M = []
    average_bit_rate_flex_rate_per_M = []
    minimum_snr_flex_per_M = []
    maximum_snr_flex_per_M = []
    average_snr_flex_per_M = []

    connections_shannon_per_M = []
    number_connections_shannon_per_M = []
    number_blocking_events_shannon_per_M = []
    capacities_shannon_per_M = []
    average_bit_rate_shannon_per_M = []
    minimum_snr_shannon_per_M = []
    maximum_snr_shannon_per_M = []
    average_snr_shannon_per_M = []

    average_average_snr_fixed = 0
    average_max_snr_fixed = 0
    average_min_snr_fixed = 0
    average_total_capacity_fixed = 0
    average_average_bit_rate_fixed = 0
    average_min_bit_rate_fixed = 0
    average_max_bit_rate_fixed = 0
    average_blocking_events_fixed = 0
    average_successful_connections_fixed = 0
    average_congestion_fixed = 0
    average_average_latency_fixed = 0
    average_max_latency_fixed = 0
    average_min_latency_fixed = 0

    average_average_snr_flex = 0
    average_max_snr_flex = 0
    average_min_snr_flex = 0
    average_total_capacity_flex = 0
    average_average_bit_rate_flex = 0
    average_min_bit_rate_flex = 0
    average_max_bit_rate_flex = 0
    average_blocking_events_flex = 0
    average_successful_connections_flex = 0
    average_congestion_flex = 0
    average_average_latency_flex = 0
    average_max_latency_flex = 0
    average_min_latency_flex = 0

    average_average_snr_shannon = 0
    average_max_snr_shannon = 0
    average_min_snr_shannon = 0
    average_total_capacity_shannon = 0
    average_average_bit_rate_shannon = 0
    average_min_bit_rate_shannon = 0
    average_max_bit_rate_shannon = 0
    average_blocking_events_shannon = 0
    average_successful_connections_shannon = 0
    average_congestion_shannon = 0
    average_average_latency_shannon = 0
    average_max_latency_shannon = 0
    average_min_latency_shannon = 0

    M_list = [16 for i in range(0, 100)]
    random_example_setter = random.randint(0, len(M_list) - 1)
    for counter, M in enumerate(M_list):
        if draw_my_plots and counter == random_example_setter:  # save these plots just for the first iteration
            snr_string = '10d1_snr_M' + str(M)
            lat_string = '10d1_lat_M' + str(M)
            br_string = '10d1_br_M' + str(M)
            in_out_distr_string = '10d1_inout_M' + str(M)
            line_occupation_string = '10d1_line_occupation_M' + str(M)

            outp_plot_snr = out_directory / 'Point1' / snr_string
            outp_plot_lat = out_directory / 'Point1' / lat_string
            outp_plot_br = out_directory / 'Point1' / br_string
            outp_plot_inout = out_directory / 'Point1' / in_out_distr_string
            outp_plot_lineocc = out_directory / 'Point1' / line_occupation_string

        traffic_matrix_fixed = utls.generate_traffic_matrix(network_fixed, M)
        [connection_tm_fixed, saturation_fixed] = network_fixed.deploy_traffic_matrix(traffic_matrix_fixed)
        [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_fixed)
        [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_fixed)
        [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_fixed)
        [_, congestion] = utls.average_line_occupation(network_fixed)
        [avg_lat, max_lat, min_lat] = utls.compute_avg_max_min_latency(connection_tm_fixed)
        print()
        print('FIXED TRANSCEIVER STRATEGY')
        print('-------------------------------')
        print('Simulation number', counter + 1, 'with M =', M)
        print('-------------------------------')
        if verbose:
            if saturation_fixed:
                print('Fixed network for M =', M, ': network was saturated.')
            elif not saturation_fixed:
                print('Fixed network for M =', M, ': traffic matrix was saturated')
        connections_fixed_per_M.append(successful_connections + blocking_events)
        number_connections_fixed_rate_per_M.append(successful_connections)
        number_blocking_events_fixed_rate_per_M.append(blocking_events)
        capacities_fixed_rate_per_M.append(total_capacity)
        average_bit_rate_fixed_rate_per_M.append(avg_bit_rate)
        average_snr_fixed_per_M.append(avg_snr)
        maximum_snr_fixed_per_M.append(max_snr)
        minimum_snr_fixed_per_M.append(min_snr)
        percentage_blocking_events = blocking_events / (blocking_events + successful_connections) * 100
        if verbose:
            print('Number of successful connections:', successful_connections)
            print('Number of blocking events:', blocking_events)
            print('Percentage of blocking events =', percentage_blocking_events, '%')
            print('Total capacity allocated =', total_capacity / 1000, 'Tbps')
            print('Average bit rate =', avg_bit_rate, 'Gbps')
            print('Average SNR =', avg_snr, 'dB')
        average_average_snr_fixed = (average_average_snr_fixed * counter + avg_snr) / (counter + 1)
        average_min_snr_fixed = (average_min_snr_fixed * counter + min_snr) / (counter + 1)
        average_max_snr_fixed = (average_max_snr_fixed * counter + max_snr) / (counter + 1)
        average_total_capacity_fixed = (average_total_capacity_fixed * counter + total_capacity) / (counter + 1)
        average_average_bit_rate_fixed = (average_average_bit_rate_fixed * counter + avg_bit_rate) / (counter + 1)
        average_min_bit_rate_fixed = (average_min_bit_rate_fixed * counter + min_br) / (counter + 1)
        average_max_bit_rate_fixed = (average_max_bit_rate_fixed * counter + max_br) / (counter + 1)
        average_blocking_events_fixed = (average_blocking_events_fixed * counter + blocking_events) / (counter + 1)
        average_successful_connections_fixed = (average_successful_connections_fixed * counter + successful_connections) / (counter + 1)
        average_congestion_fixed = (average_congestion_fixed * counter + congestion) / (counter + 1)
        average_average_latency_fixed = (average_average_latency_fixed * counter + avg_lat) / (counter + 1)
        average_max_latency_fixed = (average_max_latency_fixed * counter + max_lat) / (counter + 1)
        average_min_latency_fixed = (average_min_latency_fixed * counter + min_lat) / (counter + 1)
        print('Average average SNR at run', counter + 1, '=', average_average_snr_fixed, 'dB')
        print('Average minimum SNR at run', counter + 1, '=', average_min_snr_fixed, 'dB')
        print('Average maximum SNR at run', counter + 1, '=', average_max_snr_fixed, 'dB')
        print('Average capacity at run', counter + 1, '=', average_total_capacity_fixed / 1000, 'Tbps')
        print('Average average bit rate at run', counter + 1, '=', average_average_bit_rate_fixed, 'Gbps')
        print('Average minimum bit rate at run', counter + 1, '=', average_min_bit_rate_fixed, 'Gbps')
        print('Average maximum bit rate at run', counter + 1, '=', average_max_bit_rate_fixed, 'Gbps')
        print('Average average latency at run', counter + 1, '=', average_average_latency_fixed * 1e3, 'ms')
        print('Average minimum latency at run', counter + 1, '=', average_min_latency_fixed * 1e3, 'ms')
        print('Average maximum latency at run', counter + 1, '=', average_max_latency_fixed * 1e3, 'ms')
        print('Average number of blocking events at run', counter + 1, '=', average_blocking_events_fixed)
        print('Average number of successful connections at run', counter + 1, '=', average_successful_connections_fixed)
        print('Average wavelength congestion at run', counter + 1, '=', average_congestion_fixed, '%')

        traffic_matrix_flex = utls.generate_traffic_matrix(network_flex, M)
        [connection_tm_flex, saturation_flex] = network_flex.deploy_traffic_matrix(traffic_matrix_flex)
        [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_flex)
        [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_flex)
        [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_flex)
        [_, congestion] = utls.average_line_occupation(network_flex)
        [avg_lat, max_lat, min_lat] = utls.compute_avg_max_min_latency(connection_tm_flex)
        print()
        print('FLEXIBLE TRANSCEIVER STRATEGY')
        print('-------------------------------')
        print('Simulation number', counter + 1, 'with M =', M)
        print('-------------------------------')
        if verbose:
            if saturation_flex:
                print('Flexible network for M =', M, ': network was saturated.')
            elif not saturation_flex:
                print('Flexible network for M =', M, ': traffic matrix was saturated')
        connections_flex_per_M.append(successful_connections + blocking_events)
        number_connections_flex_rate_per_M.append(successful_connections)
        number_blocking_events_flex_rate_per_M.append(blocking_events)
        capacities_flex_rate_per_M.append(total_capacity)
        average_bit_rate_flex_rate_per_M.append(avg_bit_rate)
        average_snr_flex_per_M.append(avg_snr)
        maximum_snr_flex_per_M.append(max_snr)
        minimum_snr_flex_per_M.append(min_snr)
        percentage_blocking_events = blocking_events / (blocking_events + successful_connections) * 100
        if verbose:
            print('Number of successful connections:', successful_connections)
            print('Number of blocking events:', blocking_events)
            print('Percentage of blocking events =', percentage_blocking_events, '%')
            print('Total capacity allocated =', total_capacity / 1000, 'Tbps')
            print('Average bit rate =', avg_bit_rate, 'Gbps')
            print('Average SNR =', avg_snr, 'dB')
        average_average_snr_flex = (average_average_snr_flex * counter + avg_snr) / (counter + 1)
        average_min_snr_flex = (average_min_snr_flex * counter + min_snr) / (counter + 1)
        average_max_snr_flex = (average_max_snr_flex * counter + max_snr) / (counter + 1)
        average_total_capacity_flex = (average_total_capacity_flex * counter + total_capacity) / (counter + 1)
        average_average_bit_rate_flex = (average_average_bit_rate_flex * counter + avg_bit_rate) / (counter + 1)
        average_min_bit_rate_flex = (average_min_bit_rate_flex * counter + min_br) / (counter + 1)
        average_max_bit_rate_flex = (average_max_bit_rate_flex * counter + max_br) / (counter + 1)
        average_blocking_events_flex = (average_blocking_events_flex * counter + blocking_events) / (counter + 1)
        average_successful_connections_flex = (average_successful_connections_flex * counter + successful_connections) / (counter + 1)
        average_congestion_flex = (average_congestion_flex * counter + congestion) / (counter + 1)
        average_average_latency_flex = (average_average_latency_flex * counter + avg_lat) / (counter + 1)
        average_max_latency_flex = (average_max_latency_flex * counter + max_lat) / (counter + 1)
        average_min_latency_flex = (average_min_latency_flex * counter + min_lat) / (counter + 1)
        print('Average average SNR at run', counter + 1, '=', average_average_snr_flex, 'dB')
        print('Average minimum SNR at run', counter + 1, '=', average_min_snr_flex, 'dB')
        print('Average maximum SNR at run', counter + 1, '=', average_max_snr_flex, 'dB')
        print('Average capacity at run', counter + 1, '=', average_total_capacity_flex / 1000, 'Tbps')
        print('Average average bit rate at run', counter + 1, '=', average_average_bit_rate_flex, 'Gbps')
        print('Average minimum bit rate at run', counter + 1, '=', average_min_bit_rate_flex, 'Gbps')
        print('Average maximum bit rate at run', counter + 1, '=', average_max_bit_rate_flex, 'Gbps')
        print('Average average latency at run', counter + 1, '=', average_average_latency_flex * 1e3, 'ms')
        print('Average minimum latency at run', counter + 1, '=', average_min_latency_flex * 1e3, 'ms')
        print('Average maximum latency at run', counter + 1, '=', average_max_latency_flex * 1e3, 'ms')
        print('Average number of blocking events at run', counter + 1, '=', average_blocking_events_flex)
        print('Average number of successful connections at run', counter + 1, '=', average_successful_connections_flex)
        print('Average wavelength congestion at run', counter + 1, '=', average_congestion_flex, '%')

        traffic_matrix_shan = utls.generate_traffic_matrix(network_shan, M)
        [connection_tm_shan, saturation_shan] = network_shan.deploy_traffic_matrix(traffic_matrix_shan)
        [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_shan)
        [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_shan)
        [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_shan)
        [_, congestion] = utls.average_line_occupation(network_shan)
        [avg_lat, max_lat, min_lat] = utls.compute_avg_max_min_latency(connection_tm_shan)
        print()
        print('SHANNON TRANSCEIVER STRATEGY')
        print('-------------------------------')
        print('Simulation number', counter + 1, 'with M =', M)
        print('-------------------------------')
        if verbose:
            if saturation_shan:
                print('Shannon network for M =', M, ': network was saturated.')
            elif not saturation_shan:
                print('Shannon network for M =', M, ': traffic matrix was saturated')
        connections_shannon_per_M.append(successful_connections + blocking_events)
        number_connections_shannon_per_M.append(successful_connections)
        number_blocking_events_shannon_per_M.append(blocking_events)
        capacities_shannon_per_M.append(total_capacity)
        average_bit_rate_shannon_per_M.append(avg_bit_rate)
        average_snr_shannon_per_M.append(avg_snr)
        maximum_snr_shannon_per_M.append(max_snr)
        minimum_snr_shannon_per_M.append(min_snr)
        percentage_blocking_events = blocking_events / (blocking_events + successful_connections) * 100
        if verbose:
            print('Number of successful connections:', successful_connections)
            print('Number of blocking events:', blocking_events)
            print('Percentage of blocking events =', percentage_blocking_events, '%')
            print('Total capacity allocated =', total_capacity / 1000, 'Tbps')
            print('Average bit rate =', avg_bit_rate, 'Gbps')
            print('Average SNR =', avg_snr, 'dB')
        average_average_snr_shannon = (average_average_snr_shannon * counter + avg_snr) / (counter + 1)
        average_min_snr_shannon = (average_min_snr_shannon * counter + min_snr) / (counter + 1)
        average_max_snr_shannon = (average_max_snr_shannon * counter + max_snr) / (counter + 1)
        average_total_capacity_shannon = (average_total_capacity_shannon * counter + total_capacity) / (counter + 1)
        average_average_bit_rate_shannon = (average_average_bit_rate_shannon * counter + avg_bit_rate) / (counter + 1)
        average_min_bit_rate_shannon = (average_min_bit_rate_shannon * counter + min_br) / (counter + 1)
        average_max_bit_rate_shannon = (average_max_bit_rate_shannon * counter + max_br) / (counter + 1)
        average_blocking_events_shannon = (average_blocking_events_shannon * counter + blocking_events) / (counter + 1)
        average_successful_connections_shannon = (average_successful_connections_shannon * counter + successful_connections) / (counter + 1)
        average_congestion_shannon = (average_congestion_shannon * counter + congestion) / (counter + 1)
        average_average_latency_shannon = (average_average_latency_shannon * counter + avg_lat) / (counter + 1)
        average_max_latency_shannon = (average_max_latency_shannon * counter + max_lat) / (counter + 1)
        average_min_latency_shannon = (average_min_latency_shannon * counter + min_lat) / (counter + 1)
        print('Average average SNR at run', counter + 1, '=', average_average_snr_shannon, 'dB')
        print('Average minimum SNR at run', counter + 1, '=', average_min_snr_shannon, 'dB')
        print('Average maximum SNR at run', counter + 1, '=', average_max_snr_shannon, 'dB')
        print('Average capacity at run', counter + 1, '=', average_total_capacity_shannon / 1000, 'Tbps')
        print('Average average bit rate at run', counter + 1, '=', average_average_bit_rate_shannon, 'Gbps')
        print('Average minimum bit rate at run', counter + 1, '=', average_min_bit_rate_shannon, 'Gbps')
        print('Average maximum bit rate at run', counter + 1, '=', average_max_bit_rate_shannon, 'Gbps')
        print('Average average latency at run', counter + 1, '=', average_average_latency_shannon * 1e3, 'ms')
        print('Average minimum latency at run', counter + 1, '=', average_min_latency_shannon * 1e3, 'ms')
        print('Average maximum latency at run', counter + 1, '=', average_max_latency_shannon * 1e3, 'ms')
        print('Average number of blocking events at run', counter + 1, '=', average_blocking_events_shannon)
        print('Average number of successful connections at run', counter + 1, '=', average_successful_connections_shannon)
        print('Average wavelength congestion at run', counter + 1, '=', average_congestion_shannon, '%')
        print()

        if draw_my_plots and counter == random_example_setter:
            lat_fixed = [connection.latency for connection in connection_tm_fixed]
            lat_flex = [connection.latency for connection in connection_tm_flex]
            lat_shan = [connection.latency for connection in connection_tm_shan]
            lat_list = [lat_fixed, lat_flex, lat_shan]

            snr_fixed = [connection.snr for connection in connection_tm_fixed if connection.snr != 0]
            snr_flex = [connection.snr for connection in connection_tm_flex if connection.snr != 0]
            snr_shan = [connection.snr for connection in connection_tm_shan if connection.snr != 0]
            snr_list = [snr_fixed, snr_flex, snr_shan]

            br_fixed = [connection.bit_rate for connection in connection_tm_fixed]
            br_flex = [connection.bit_rate for connection in connection_tm_flex]
            br_shan = [connection.bit_rate for connection in connection_tm_shan]
            br_list = [br_fixed, br_flex, br_shan]

            [list_pairs, in_out_distr_fixed] = utls.compute_in_out_node_distribution(network_fixed, connection_tm_fixed)
            [_, in_out_distr_flex] = utls.compute_in_out_node_distribution(network_flex, connection_tm_flex)
            [_, in_out_distr_shannon] = utls.compute_in_out_node_distribution(network_shan, connection_tm_shan)
            in_out_distr_list = [in_out_distr_fixed, in_out_distr_flex, in_out_distr_shannon]

            [list_lines, line_occupation_fixed] = utls.count_line_usage(network_fixed)
            [_, line_occupation_flex] = utls.count_line_usage(network_flex)
            [_, line_occupation_shannon] = utls.count_line_usage(network_shan)
            line_occupation_list = [line_occupation_fixed, line_occupation_flex, line_occupation_shannon]

            utls.plot_histogram(1, snr_list, 20, 'k', ['m', 'b', 'tab:orange'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                                'Single Matrix analysis - SNR distribution, M = ' + str(M), 'Number of results',
                                'SNR, dB')
            if save_my_figure:
                plt.savefig(outp_plot_snr)

            utls.plot_histogram(2, lat_list, 20, 'k', ['m', 'b', 'tab:orange'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                                'Single Matrix analysis - Latency, M = ' + str(M), 'Number of results', 'Latency, s')
            if save_my_figure:
                plt.savefig(outp_plot_lat)

            utls.plot_histogram(3, br_list, 15, 'k', ['m', 'b', 'tab:orange'], ['Fixed rate', 'Flexible rate', 'Shannon rate'],
                                'Single Matrix analysis - Bit rate, M = ' + str(M), 'Number of results',
                                'Bit rate, Gbps')
            if save_my_figure:
                plt.savefig(outp_plot_br)

            utls.plot_bar(figure_num=181, list_data=in_out_distr_list, bbox_to_anchor=(0.5, -0.35), bottom=0.25,
                          loc='lower center', edge_color='k', color=['m', 'b', 'tab:orange'],
                          label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='',
                          ylabel='Number of occurrences',
                          title='Single Matrix analysis - Input and output node pairs, M = ' + str(M), myalpha=1,
                          x_ticks=list_pairs)
            if save_my_figure:
                plt.savefig(outp_plot_inout)

            utls.plot_bar(figure_num=182, list_data=line_occupation_list, bbox_to_anchor=(0.5, -0.35), bottom=0.25,
                          loc='lower center', edge_color='k', color=['m', 'b', 'tab:orange'],
                          label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='',
                          ylabel='Number of occurrences',
                          title='Single Matrix analysis - Line occupation, M = ' + str(M), myalpha=1,
                          x_ticks=list_lines)
            if save_my_figure:
                plt.savefig(outp_plot_lineocc)

        utls.free_lines_and_switch_matrix(file_fixed, network_fixed)
        utls.free_lines_and_switch_matrix(file_flex, network_flex)
        utls.free_lines_and_switch_matrix(file_shannon, network_shan)

    if draw_my_plots:
        string_for_titles = 'M = ' + str(M_list[0]) + ' and ' + str(len(M_list)) + ' runs'
        utls.plot_bar(figure_num=4, list_data=[[average_average_snr_fixed],
                                               [average_average_snr_flex],
                                               [average_average_snr_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='SNR, dB',
                      title='Single Matrix analysis - Average SNR, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_avg_snr)

        utls.plot_bar(figure_num=5, list_data=[[average_min_snr_fixed],
                                               [average_min_snr_flex],
                                               [average_min_snr_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='SNR, dB',
                      title='Single Matrix analysis - Minimum SNR, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_min_snr)

        utls.plot_bar(figure_num=6, list_data=[[average_max_snr_fixed],
                                               [average_max_snr_flex],
                                               [average_max_snr_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='SNR, dB',
                      title='Single Matrix analysis - Maximum SNR, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_max_snr)

        utls.plot_bar(figure_num=7, list_data=[[average_total_capacity_fixed / 1000],
                                               [average_total_capacity_flex / 1000],
                                               [average_total_capacity_shannon / 1000]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='Capacity, Tbps',
                      title='Single Matrix analysis - Total capacity, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_capacity)

        utls.plot_bar(figure_num=8, list_data=[[average_average_bit_rate_fixed], [average_average_bit_rate_flex],
                                               [average_average_bit_rate_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='Bit rate, Gbps',
                      title='Single Matrix analysis - Average bit rate, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_avg_br)

        utls.plot_bar(figure_num=9, list_data=[[average_min_bit_rate_fixed],
                                               [average_min_bit_rate_flex],
                                               [average_min_bit_rate_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='Bit rate, Gbps',
                      title='Single Matrix analysis - Minimum bit rate, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_min_br)

        utls.plot_bar(figure_num=10, list_data=[[average_max_bit_rate_fixed],
                                                [average_max_bit_rate_flex],
                                                [average_max_bit_rate_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='Bit rate, Gbps',
                      title='Single Matrix analysis - Maximum bit rate, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_max_br)

        utls.plot_bar(figure_num=11, list_data=[[average_blocking_events_fixed],
                                                [average_blocking_events_flex],
                                                [average_blocking_events_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='Number of blocking events',
                      title='Single Matrix analysis - Blocking Events, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_block_events)

        utls.plot_bar(figure_num=12, list_data=[[average_successful_connections_fixed],
                                                [average_successful_connections_flex],
                                                [average_successful_connections_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='Number of connections',
                      title='Single Matrix analysis - Successful connections, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_connections)

        utls.plot_bar(figure_num=13, list_data=[[average_congestion_fixed],
                                                [average_congestion_flex],
                                                [average_congestion_shannon]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='', ylabel='Congestion, %',
                      title='Single Matrix analysis - Wavelength congestion, ' + string_for_titles, myalpha=1)
        if save_my_figure:
            plt.savefig(outp_plot_congestion)

        plt.show()

# ----------------- POINT 2 (Network Congestion) -----------------
if do_simulations[1]:
    connections_fixed_per_M = []
    number_connections_fixed_rate_per_M = []
    number_blocking_events_fixed_rate_per_M = []
    capacities_fixed_rate_per_M = []
    average_bit_rate_fixed_rate_per_M = []
    minimum_snr_fixed_per_M = []
    maximum_snr_fixed_per_M = []
    average_snr_fixed_per_M = []
    minimum_bit_rate_fixed_per_M = []
    maximum_bit_rate_fixed_per_M = []
    congestion_fixed_per_M = []

    connections_flex_per_M = []
    number_connections_flex_rate_per_M = []
    number_blocking_events_flex_rate_per_M = []
    capacities_flex_rate_per_M = []
    average_bit_rate_flex_rate_per_M = []
    minimum_snr_flex_per_M = []
    maximum_snr_flex_per_M = []
    average_snr_flex_per_M = []
    minimum_bit_rate_flex_per_M = []
    maximum_bit_rate_flex_per_M = []
    congestion_flex_per_M = []

    connections_shannon_per_M = []
    number_connections_shannon_per_M = []
    number_blocking_events_shannon_per_M = []
    capacities_shannon_per_M = []
    average_bit_rate_shannon_per_M = []
    minimum_snr_shannon_per_M = []
    maximum_snr_shannon_per_M = []
    average_snr_shannon_per_M = []
    minimum_bit_rate_shannon_per_M = []
    maximum_bit_rate_shannon_per_M = []
    congestion_shannon_per_M = []

    M_list = []
    for M in range(1, 10, 2):  # range(10, 51, 10):
        M_list.append(M)
        M_for_loop = [M for i in range(0, 20)]
        print('----------------------')
        print('Simulation with M =', M)
        print('----------------------')

        average_average_snr_fixed = 0
        average_max_snr_fixed = 0
        average_min_snr_fixed = 0
        average_total_capacity_fixed = 0
        average_average_bit_rate_fixed = 0
        average_min_bit_rate_fixed = 0
        average_max_bit_rate_fixed = 0
        average_blocking_events_fixed = 0
        average_successful_connections_fixed = 0
        average_congestion_fixed = 0

        average_average_snr_flex = 0
        average_max_snr_flex = 0
        average_min_snr_flex = 0
        average_total_capacity_flex = 0
        average_average_bit_rate_flex = 0
        average_min_bit_rate_flex = 0
        average_max_bit_rate_flex = 0
        average_blocking_events_flex = 0
        average_successful_connections_flex = 0
        average_congestion_flex = 0

        average_average_snr_shannon = 0
        average_max_snr_shannon = 0
        average_min_snr_shannon = 0
        average_total_capacity_shannon = 0
        average_average_bit_rate_shannon = 0
        average_min_bit_rate_shannon = 0
        average_max_bit_rate_shannon = 0
        average_blocking_events_shannon = 0
        average_successful_connections_shannon = 0
        average_congestion_shannon = 0
        print('Computing', end=" ")
        for counter, M_sm in enumerate(M_for_loop):
            print('.', end=" ")
            # FIXED STRATEGY
            traffic_matrix_fixed = utls.generate_traffic_matrix(network_fixed, M_sm)
            [connection_tm_fixed, saturation_fixed] = network_fixed.deploy_traffic_matrix(traffic_matrix_fixed)
            [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_fixed)
            [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_fixed)
            [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_fixed)
            [_, congestion] = utls.average_line_occupation(network_fixed)
            # update averages
            average_average_snr_fixed = (average_average_snr_fixed * counter + avg_snr) / (counter + 1)
            average_min_snr_fixed = (average_min_snr_fixed * counter + min_snr) / (counter + 1)
            average_max_snr_fixed = (average_max_snr_fixed * counter + max_snr) / (counter + 1)
            average_total_capacity_fixed = (average_total_capacity_fixed * counter + total_capacity) / (counter + 1)
            average_average_bit_rate_fixed = (average_average_bit_rate_fixed * counter + avg_bit_rate) / (counter + 1)
            average_min_bit_rate_fixed = (average_min_bit_rate_fixed * counter + min_br) / (counter + 1)
            average_max_bit_rate_fixed = (average_max_bit_rate_fixed * counter + max_br) / (counter + 1)
            average_blocking_events_fixed = (average_blocking_events_fixed * counter + blocking_events) / (counter + 1)
            average_successful_connections_fixed = (average_successful_connections_fixed * counter
                                                    + successful_connections) / (counter + 1)
            average_congestion_fixed = (average_congestion_fixed * counter + congestion) / (counter + 1)
            # reset lines for next deployment
            utls.free_lines_and_switch_matrix(file_fixed, network_fixed)

            # FLEXIBLE STRATEGY
            traffic_matrix_flex = utls.generate_traffic_matrix(network_flex, M_sm)
            [connection_tm_flex, saturation_flex] = network_flex.deploy_traffic_matrix(traffic_matrix_flex)
            [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_flex)
            [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_flex)
            [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_flex)
            [_, congestion] = utls.average_line_occupation(network_flex)
            # update averages
            average_average_snr_flex = (average_average_snr_flex * counter + avg_snr) / (counter + 1)
            average_min_snr_flex = (average_min_snr_flex * counter + min_snr) / (counter + 1)
            average_max_snr_flex = (average_max_snr_flex * counter + max_snr) / (counter + 1)
            average_total_capacity_flex = (average_total_capacity_flex * counter + total_capacity) / (counter + 1)
            average_average_bit_rate_flex = (average_average_bit_rate_flex * counter + avg_bit_rate) / (counter + 1)
            average_min_bit_rate_flex = (average_min_bit_rate_flex * counter + min_br) / (counter + 1)
            average_max_bit_rate_flex = (average_max_bit_rate_flex * counter + max_br) / (counter + 1)
            average_blocking_events_flex = (average_blocking_events_flex * counter + blocking_events) / (counter + 1)
            average_successful_connections_flex = (average_successful_connections_flex * counter +
                                                   successful_connections) / (counter + 1)
            average_congestion_flex = (average_congestion_flex * counter + congestion) / (counter + 1)
            # reset lines for next deployment
            utls.free_lines_and_switch_matrix(file_flex, network_flex)

            # SHANNON STRATEGY
            traffic_matrix_shan = utls.generate_traffic_matrix(network_shan, M_sm)
            [connection_tm_shan, saturation_shan] = network_shan.deploy_traffic_matrix(traffic_matrix_shan)
            [successful_connections, blocking_events] = utls.compute_successful_blocking_events(connection_tm_shan)
            [total_capacity, avg_bit_rate, max_br, min_br] = utls.compute_network_capacity_and_avg_bit_rate(connection_tm_shan)
            [avg_snr, max_snr, min_snr] = utls.compute_average_max_min_snr(connection_tm_shan)
            [_, congestion] = utls.average_line_occupation(network_shan)
            # update averages
            average_average_snr_shannon = (average_average_snr_shannon * counter + avg_snr) / (counter + 1)
            average_min_snr_shannon = (average_min_snr_shannon * counter + min_snr) / (counter + 1)
            average_max_snr_shannon = (average_max_snr_shannon * counter + max_snr) / (counter + 1)
            average_total_capacity_shannon = (average_total_capacity_shannon * counter + total_capacity) / (counter + 1)
            average_average_bit_rate_shannon = (average_average_bit_rate_shannon * counter + avg_bit_rate) / (
                        counter + 1)
            average_min_bit_rate_shannon = (average_min_bit_rate_shannon * counter + min_br) / (counter + 1)
            average_max_bit_rate_shannon = (average_max_bit_rate_shannon * counter + max_br) / (counter + 1)
            average_blocking_events_shannon = (average_blocking_events_shannon * counter + blocking_events) / (
                        counter + 1)
            average_successful_connections_shannon = (average_successful_connections_shannon * counter +
                                                      successful_connections) / (counter + 1)
            average_congestion_shannon = (average_congestion_shannon * counter + congestion) / (counter + 1)
            # reset lines for next deployment
            utls.free_lines_and_switch_matrix(file_shannon, network_shan)

        connections_fixed_per_M.append(average_successful_connections_fixed + average_blocking_events_fixed)
        number_connections_fixed_rate_per_M.append(average_successful_connections_fixed)
        number_blocking_events_fixed_rate_per_M.append(average_blocking_events_fixed)
        capacities_fixed_rate_per_M.append(average_total_capacity_fixed)
        average_bit_rate_fixed_rate_per_M.append(average_average_bit_rate_fixed)
        minimum_bit_rate_fixed_per_M.append(average_min_bit_rate_fixed)
        maximum_bit_rate_fixed_per_M.append(average_max_bit_rate_fixed)
        average_snr_fixed_per_M.append(average_average_snr_fixed)
        maximum_snr_fixed_per_M.append(average_max_snr_fixed)
        minimum_snr_fixed_per_M.append(average_min_snr_fixed)
        congestion_fixed_per_M.append(average_congestion_fixed)
        print()
        print('\nFIXED')
        print('Number of successful connections:', average_successful_connections_fixed, '.')
        print('Number of blocking events:', average_blocking_events_fixed, '.')
        print('Total capacity allocated =', average_total_capacity_fixed / 1000, 'Tbps.')
        print('Average bit rate =', average_average_bit_rate_fixed, 'Gbps.')
        print('Average SNR =', average_average_snr_fixed, 'dB')
        print('Minimum SNR =', average_min_snr_fixed, 'dB')
        print('Maximum SNR =', average_max_snr_fixed, 'dB')
        print('Wavelength congestion =', average_congestion_fixed, '%')
        print()

        connections_flex_per_M.append(average_successful_connections_flex + average_blocking_events_flex)
        number_connections_flex_rate_per_M.append(average_successful_connections_flex)
        number_blocking_events_flex_rate_per_M.append(average_blocking_events_flex)
        capacities_flex_rate_per_M.append(average_total_capacity_flex)
        average_bit_rate_flex_rate_per_M.append(average_average_bit_rate_flex)
        minimum_bit_rate_flex_per_M.append(average_min_bit_rate_flex)
        maximum_bit_rate_flex_per_M.append(average_max_bit_rate_flex)
        average_snr_flex_per_M.append(average_average_snr_flex)
        maximum_snr_flex_per_M.append(average_max_snr_flex)
        minimum_snr_flex_per_M.append(average_min_snr_flex)
        congestion_flex_per_M.append(average_congestion_flex)
        print('FLEXIBLE')
        print('Number of successful connections:', average_successful_connections_flex, '.')
        print('Number of blocking events:', average_blocking_events_flex, '.')
        print('Total capacity allocated =', average_total_capacity_flex / 1000, 'Tbps.')
        print('Average bit rate =', average_average_bit_rate_flex, 'Gbps.')
        print('Average SNR =', average_average_snr_flex, 'dB')
        print('Minimum SNR =', average_min_snr_flex, 'dB')
        print('Maximum SNR =', average_max_snr_flex, 'dB')
        print('Wavelength congestion =', average_congestion_flex, '%')
        print()

        connections_shannon_per_M.append(average_successful_connections_shannon + average_blocking_events_shannon)
        number_connections_shannon_per_M.append(average_successful_connections_shannon)
        number_blocking_events_shannon_per_M.append(average_blocking_events_shannon)
        capacities_shannon_per_M.append(average_total_capacity_shannon)
        average_bit_rate_shannon_per_M.append(average_average_bit_rate_shannon)
        minimum_bit_rate_shannon_per_M.append(average_min_bit_rate_shannon)
        maximum_bit_rate_shannon_per_M.append(average_max_bit_rate_shannon)
        average_snr_shannon_per_M.append(average_average_snr_shannon)
        maximum_snr_shannon_per_M.append(average_max_snr_shannon)
        minimum_snr_shannon_per_M.append(average_min_snr_shannon)
        congestion_shannon_per_M.append(average_congestion_shannon)
        print('SHANNON')
        print('Number of successful connections:', average_successful_connections_shannon, '.')
        print('Number of blocking events:', average_blocking_events_shannon, '.')
        print('Total capacity allocated =', average_total_capacity_shannon / 1000, 'Tbps.')
        print('Average bit rate =', average_average_bit_rate_shannon, 'Gbps.')
        print('Average SNR =', average_average_snr_shannon, 'dB')
        print('Minimum SNR =', average_min_snr_shannon, 'dB')
        print('Maximum SNR =', average_max_snr_shannon, 'dB')
        print('Wavelength congestion =', average_congestion_shannon, '%')
        print()
        print()

    if draw_my_plots:
        # output path strings
        from_to_M_string = str(M_list[0])+'_to_' + str(M_list[len(M_list)-1])
        outp_plot_avg_snr = out_directory / 'Point2' / ('10d2_average_snr_M_' + from_to_M_string)
        outp_plot_min_snr = out_directory / 'Point2' / ('10d2_min_snr_M' + from_to_M_string)
        outp_plot_max_snr = out_directory / 'Point2' / ('10d2_max_snr_M' + from_to_M_string)
        outp_plot_capacity = out_directory / 'Point2' / ('10d2_capacity_M' + from_to_M_string)
        outp_plot_avg_br = out_directory / 'Point2' / ('10d2_average_bit_rate_M' + from_to_M_string)
        outp_plot_min_br = out_directory / 'Point2' / ('10d2_min_bit_rate_M' + from_to_M_string)
        outp_plot_max_br = out_directory / 'Point2' / ('10d2_max_bit_rate_M' + from_to_M_string)
        outp_plot_block_events = out_directory / 'Point2' / ('10d2_blocking_events_M' + from_to_M_string)
        outp_plot_connections = out_directory / 'Point2' / ('10d2_successful_connections_M' + from_to_M_string)
        outp_plot_congestion = out_directory / 'Point2' / ('10d2_congestion' + from_to_M_string)

        string_for_titles = 'with M from ' + str(M_list[0]) + ' to ' + str(M_list[len(M_list)-1])
        utls.plot_bar(figure_num=14, list_data=[[average_snr_fixed_per_M[i] for i in range(0, len(M_list))],
                                                [average_snr_flex_per_M[i] for i in range(0, len(M_list))],
                                                [average_snr_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='SNR, dB',
                      title='Network congestion analysis - Average SNR, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_avg_snr)

        utls.plot_bar(figure_num=15, list_data=[[minimum_snr_fixed_per_M[i] for i in range(0, len(M_list))],
                                                [minimum_snr_flex_per_M[i] for i in range(0, len(M_list))],
                                                [minimum_snr_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='SNR, dB',
                      title='Network congestion analysis - Minimum SNR, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_min_snr)

        utls.plot_bar(figure_num=16, list_data=[[maximum_snr_fixed_per_M[i] for i in range(0, len(M_list))],
                                                [maximum_snr_flex_per_M[i] for i in range(0, len(M_list))],
                                                [maximum_snr_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='SNR, dB',
                      title='Network congestion analysis - Maximum SNR, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_max_snr)

        utls.plot_bar(figure_num=17, list_data=[[capacities_fixed_rate_per_M[i] / 1000 for i in range(0, len(M_list))],
                                                [capacities_flex_rate_per_M[i] / 1000 for i in range(0, len(M_list))],
                                                [capacities_shannon_per_M[i] / 1000 for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='Capacity, Tbps',
                      title='Network congestion analysis - Total capacity, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list], y_range=[0, 90])
        if save_my_figure:
            plt.savefig(outp_plot_capacity)

        utls.plot_bar(figure_num=18, list_data=[[average_bit_rate_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                                [average_bit_rate_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                                [average_bit_rate_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='Bit rate, Gbps',
                      title='Network congestion analysis - Average bit rate, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_avg_br)

        utls.plot_bar(figure_num=19, list_data=[[minimum_bit_rate_fixed_per_M[i] for i in range(0, len(M_list))],
                                                [minimum_bit_rate_flex_per_M[i] for i in range(0, len(M_list))],
                                                [minimum_bit_rate_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='Bit rate, Gbps',
                      title='Network congestion analysis - Minimum bit rate, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_min_br)

        utls.plot_bar(figure_num=20, list_data=[[maximum_bit_rate_fixed_per_M[i] for i in range(0, len(M_list))],
                                                [maximum_bit_rate_flex_per_M[i] for i in range(0, len(M_list))],
                                                [maximum_bit_rate_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='Bit rate, Gbps',
                      title='Network congestion analysis - Maximum bit rate, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_max_br)

        utls.plot_bar(figure_num=21, list_data=[[number_blocking_events_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                                [number_blocking_events_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                                [number_blocking_events_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='Number of blocking events',
                      title='Network congestion analysis - Blocking Events, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_block_events)

        utls.plot_bar(figure_num=22, list_data=[[number_connections_fixed_rate_per_M[i] for i in range(0, len(M_list))],
                                                [number_connections_flex_rate_per_M[i] for i in range(0, len(M_list))],
                                                [number_connections_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='Number of connections',
                      title='Network congestion analysis - Successful connections, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_connections)

        utls.plot_bar(figure_num=23, list_data=[[congestion_fixed_per_M[i] for i in range(0, len(M_list))],
                                                [congestion_flex_per_M[i] for i in range(0, len(M_list))],
                                                [congestion_shannon_per_M[i] for i in range(0, len(M_list))]],
                      bbox_to_anchor=(0.5, -0.35), bottom=0.25, loc='lower center', edge_color='k',
                      color=['m', 'b', 'tab:orange'],
                      label=['Fixed Rate', 'Flex Rate', 'Shannon Rate'], xlabel='M', ylabel='Congestion, %',
                      title='Network congestion analysis - Wavelength congestion, ' + string_for_titles, myalpha=1,
                      x_ticks=[M for M in M_list])
        if save_my_figure:
            plt.savefig(outp_plot_congestion)

        plt.show()
