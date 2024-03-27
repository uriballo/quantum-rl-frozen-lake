import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn.functional as F

dev = qml.device("lightning.qubit", wires=4)


def decimal_to_binary(_decimal, _length=4):
    bin_num = bin(int(_decimal))[2:]
    output_num = [int(item) for item in bin_num]
    if len(output_num) < _length:
        output_num = np.concatenate((np.zeros((_length - len(output_num),)), np.array(output_num)))
    else:
        output_num = np.array(output_num)
    return output_num


@qml.qnode(dev, interface='torch', diff_method='adjoint')
def circuit(inputs, weights):
    """
    :param weights: Nx4x3 matrix of weights.
    :param inputs: binary encoded state.
    """

    for idx in range(len(inputs)):
        qml.RX(np.pi * inputs[idx], wires=idx)
        qml.RZ(np.pi * inputs[idx], wires=idx)

    for W in weights:
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])

        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
        qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    return [qml.expval(qml.PauliZ(idx)) for idx in range(4)]


class HQNN(nn.Module):
    def __init__(self):
        super(HQNN, self).__init__()

        weight_shapes = {"weights": (4, 4, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        encoded_input = [decimal_to_binary(i) for i in x]

        processed_input = torch.stack([self.qlayer(torch.tensor(state)) for state in encoded_input]).float()
        processed_input = self.fc(processed_input)
        return F.leaky_relu(processed_input)



