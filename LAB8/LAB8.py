import core.elements as el
import random
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).parent.parent
in_directory = root / 'resources'
out_directory = root / 'results'

network = el.Network()
network.connect()
