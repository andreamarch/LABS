import numpy as np
import scipy.constants as consts

gain = 10 ** (16 / 10)  # 16dB in linear units
n_figure = 10 ** (3 / 10)  # 3dB in linear units
alpha_db = 0.2  # [dB/km]
alpha = 0.2 / (20 * np.log10(np.exp(1)))  # [1/km]
span_length = 80  # [km]
beta2 = 2.13e-26  # [(m * Hz^2)^-1]
gamma = 1.27e-3  # [(m W)^-1]
l_eff = 1 / (2 * alpha)  # [km]
alpha_m = 1 / (1 / alpha * 1e3)  # [1/m]
sym_rate = 32e9  # [Hz]
df = 50e9  # [Hz]

noise_bw = 12.5e9  # [Hz]
h = consts.h    # Plack's constant
f0 = 193.414e12  # [Hz]