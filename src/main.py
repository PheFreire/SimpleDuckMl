import cv2
import os
import numpy as np
from random import randint
from datasets.dataset_unpacker import DatasetUnpacker
from datasets.dataset import Dataset
import container

# -=-=-=-=-=-=-=-=-=-=-=-=-
from layers.convolutional_layer import ConvolutionalLayer
from layers.flatten_layer import FlattenLayer
from layers.dense_layer import DenseLayer
from activations.softmax_activation import SoftmaxActivation
from activations.relu_activation import ReLuActivation
from loss.cross_entropy_loss import CrossEntropyLoss
from models.model import Model

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

np.set_printoptions(precision=4, suppress=True)

ROOT = "/Users/phepato/Documents/dev/Lemon/ConvolutionalDuckMl"
DATASET = os.path.join(ROOT, "dataset")
MODELS_PATH = os.path.join(ROOT, "model")
MODEL_NAME = "duck"

EPOCHS = 20
BATCH_SIZE = 5
LEARNING_RATE = 0.001

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def get_dataset(name: str, label: int, qnt: int = 100) -> Dataset:
    path = os.path.join(DATASET, f"{name}.bin")
    unpacker = DatasetUnpacker(path)
    dataset = unpacker.unpack(label=label, qnt=qnt)
    print(
        f"[DATASET] {name}: {len(dataset.x)} imagens carregadas "
        f"(mean={dataset.x.mean():.4f}, std={dataset.x.std():.4f})"
    )
    return dataset


duck_dataset = get_dataset("duck", 0, qnt=100)
ice_dataset = get_dataset("ice_cream", 1, qnt=100)
crab_dataset = get_dataset("crab", 2, qnt=100)

datasets = [duck_dataset, ice_dataset, crab_dataset]
label_to_name = {0: "duck", 1: "ice_cream", 2: "crab"}

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if MODELS_PATH and os.listdir(MODELS_PATH):
    model = Model.load(os.path.join(MODELS_PATH, MODEL_NAME))

else:
    layers = [
        ConvolutionalLayer(
            nodes_num=4,
            kernel_shape=(5, 5, 1),
            stride=1,
            activation=ReLuActivation()
        ),
        ConvolutionalLayer(
            nodes_num=8,
            kernel_shape=(3, 3, 4),
            stride=1,
            activation=ReLuActivation()
        ),
        FlattenLayer(),
        DenseLayer(
            output_size=3,
            activation=SoftmaxActivation()
        )
    ]
    
    model = Model(
        layers=layers,
        loss=CrossEntropyLoss(),
        learning_rate=LEARNING_RATE
    )

    model.fit(
        datasets=datasets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=False
    )

    model.save(MODEL_NAME, MODELS_PATH)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def show(img):
    cv2.imshow("Preview", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test(img, model):
    pred = model.forward(img)
    label = label_to_name[int(np.argmax(pred))]
    print(f"Pred: {pred.T}")
    print(f"Classe prevista: {label}")

print("\n[TEST] Running manual Tests...\n")

while True:
    duck = duck_dataset.x[randint(0, len(duck_dataset.x)-1)]
    ice_cream = ice_dataset.x[randint(0, len(ice_dataset.x)-1)]
    crab = crab_dataset.x[randint(0, len(crab_dataset.x)-1)]

    breakpoint()

