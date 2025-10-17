from duckdi import register

from activations.relu_activation import ReLuActivation
from activations.softmax_activation import SoftmaxActivation
from layers.convolutional_layer import ConvolutionalLayer
from layers.dense_layer import DenseLayer
from layers.flatten_layer import FlattenLayer
from loss.cross_entropy_loss import CrossEntropyLoss
from loss.mse_loss import MSELoss

# Activations
register(ReLuActivation, label="relu")
register(SoftmaxActivation, label="softmax")

# Layers
register(ConvolutionalLayer, label="conv")
register(DenseLayer, label="dense")
register(FlattenLayer, label="flat")

# Loss
register(CrossEntropyLoss, label="cross_entropy")
register(MSELoss, label="mse")


