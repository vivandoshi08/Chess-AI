import chess
from networks.model import Network

net = Network()
print("Training...")
net.train(steps=100)
print("Exporting...")
net.export()