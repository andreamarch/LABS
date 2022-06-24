import numpy as np
import scipy.constants as consts

number_of_channels = 10  # number of available channels
gain = 10 ** (16 / 10)  # Gain (16dB in linear units)
n_figure = 10 ** (3 / 10)  # Noise figure (3dB in linear units)
alpha_db = 0.2  # Loss coefficient [dB/km]
alpha = 0.2 / (20 * np.log10(np.exp(1)))  # [1/km]
span_length = 80  # Length of the single fiber spans [km]
beta2 = 2.13e-26  # Beta coefficient [(m * Hz^2)^-1]
gamma = 1.27e-3  # Gamma coefficient [(m W)^-1]
l_eff = 1 / (2 * alpha)  # Effective length [km]
alpha_m = 1 / (1 / alpha * 1e3)  # [1/m]
sym_rate = 32e9  # Symbol rate [Hz]
df = 50e9  # [Hz]

noise_bw = 12.5e9  # Noise bandwidth [Hz]
h = consts.h    # Plack's constant
f0 = 193.414e12  # Reference frequency [Hz]

strategy_bit_rate = 'fixed_rate'
number_of_connections = 100

max_number_of_iterations = 200  # max number of iterations for the deploy_traffic_matrix
be_threshold = 15/100  # threshold for the number of blocking events

input_file_flag = 'exam'  # 'exam' or 'lab'
