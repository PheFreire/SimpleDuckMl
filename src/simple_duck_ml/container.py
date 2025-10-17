from duckdi import register

from simple_duck_ml.activations.relu_activation import ReLuActivation
from simple_duck_ml.activations.softmax_activation import SoftmaxActivation
from simple_duck_ml.dataset_unpacker.bin_dataset_unpacker import BinDatasetUnpacker
from simple_duck_ml.layers.convolutional_layer import ConvolutionalLayer
from simple_duck_ml.layers.dense_layer import DenseLayer
from simple_duck_ml.layers.flatten_layer import FlattenLayer
from simple_duck_ml.loss.cross_entropy_loss import CrossEntropyLoss
from simple_duck_ml.loss.mse_loss import MSELoss

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

# DatasetUnpacker
register(BinDatasetUnpacker, label="bin")
