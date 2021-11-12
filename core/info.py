#  DYNAMIC ELEMENTS

class SignalInformation:
    def __init__(self, signal_power, path, noise_power=0.0, latency=0.0):
        self.signal_power = float(signal_power)
        self.path = path
        self.noise_power = noise_power
        self.latency = latency

    def update_signal_pow(self, signal_power_increment):
        self.signal_power += signal_power_increment
        return

    def update_noise_pow(self, noise_power_increment):
        self.noise_power += noise_power_increment
        return

    def update_latency(self, latency_increment):
        self.latency += latency_increment
        return

    def update_node(self, node):
        if self.path[0] == node.label:
            self.path.remove(self.path[0])
        return self.path