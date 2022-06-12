#  DYNAMIC ELEMENTS

class SignalInformation:
    def __init__(self, path, signal_power=1.0, noise_power=0.0, latency=0.0):
        self.signal_power = float(signal_power)
        self.path = path
        self.noise_power = float(noise_power)
        self.latency = latency
        self.sym_rate = 32e9  # symbol rate in GHz
        self.df = 50e9  # df in GHz
        self.isnr = float(0.0)
        self.average_power = 0.0
        self.path_length = 0

    def update_signal_pow_average(self):
        self.average_power += self.signal_power
        if len(self.path) <= 1:
            self.average_power /= self.path_length

    def update_noise_pow(self, noise_power_increment):
        self.noise_power += noise_power_increment

    def update_isnr(self, isnr_increment):
        self.isnr += isnr_increment

    def update_latency(self, latency_increment):
        self.latency += latency_increment

    def update_node(self, node):
        if self.path[0] == node.label:
            self.path.remove(self.path[0])
        return self.path


class Lightpath(SignalInformation):
    def __init__(self, path, channel, noise_power=0.0, latency=0.0):
        super().__init__(path, noise_power=0.0, latency=0.0)
        self.channel = channel
