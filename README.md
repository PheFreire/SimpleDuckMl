# SimpleDuckML

Framework de machine learning construído do zero em Python puro com NumPy. O objetivo é ser uma referência didática e funcional: cada operação matemática que acontece durante o treinamento de uma rede neural está explícita no código, sem caixas pretas de alto nível. Você pode treinar uma CNN real enquanto lê exatamente o que acontece em cada passo.

---

## Instalação

```bash
pip install simple-duck-ml
```

---

## Exemplo rápido

```python
from simple_duck_ml import (
    Model,
    ConvolutionalLayer, FlattenLayer, DenseLayer,
    ReLuActivation, SoftmaxActivation,
    CrossEntropyLoss,
    MiniBatchBinDatasetUnpacker,
    StreamingDataSource,
)

model = Model(
    layers=[
        ConvolutionalLayer(nodes_num=8, kernel_shape=[3, 3, 1], activation=ReLuActivation()),
        FlattenLayer(),
        DenseLayer(output_size=64, activation=ReLuActivation()),
        DenseLayer(output_size=3, activation=SoftmaxActivation()),
    ],
    loss=CrossEntropyLoss(),
    learning_rate=0.01,
)

model.fit(
    sources=[
        StreamingDataSource(MiniBatchBinDatasetUnpacker("gato.bin"),    label=0, normalization=lambda x: x / 255.0),
        StreamingDataSource(MiniBatchBinDatasetUnpacker("cachorro.bin"), label=1, normalization=lambda x: x / 255.0),
        StreamingDataSource(MiniBatchBinDatasetUnpacker("passaro.bin"),  label=2, normalization=lambda x: x / 255.0),
    ],
    epochs=10,
    batch_size=32,
)

model.save(name="meu_modelo", path="./modelos")
```

---

## Arquitetura do framework

```
simple_duck_ml/
├── models/
│   └── model.py              # Loop de treino, forward, backward, update
├── layers/
│   ├── i_layer.py            # Interface de layer
│   ├── convolutional_layer.py
│   ├── dense_layer.py
│   └── flatten_layer.py
├── activations/
│   ├── i_activation.py
│   ├── relu_activation.py
│   └── softmax_activation.py
├── loss/
│   ├── i_loss.py
│   ├── cross_entropy_loss.py
│   └── mse_loss.py
└── dataset_unpacker/
    ├── i_data_source.py              # Interface de fonte de dados
    ├── in_memory_data_source.py      # Dados pré-carregados em RAM
    ├── streaming_data_source.py      # Leitura sob demanda do disco
    ├── minibatch_bin_dataset_unpacker.py
    └── bin_dataset_unpacker.py
```

---

## Fontes de dados

O método `Model.fit()` recebe uma lista de `IDataSource`. Existem duas implementações prontas:

### `InMemoryDataSource`
Para datasets pequenos que cabem na RAM. Recebe um `Dataset` já carregado.

```python
from simple_duck_ml import InMemoryDataSource, BinDatasetUnpacker

dataset = BinDatasetUnpacker("gato.bin").unpack(label=0, qnt=500)
source = InMemoryDataSource(dataset)
```

### `StreamingDataSource`
Para datasets grandes. Lê cada imagem do disco apenas quando ela é necessária durante o treino — somente `batch_size` imagens existem na RAM a qualquer momento.

```python
from simple_duck_ml import StreamingDataSource, MiniBatchBinDatasetUnpacker

source = StreamingDataSource(
    unpacker=MiniBatchBinDatasetUnpacker("gato.bin"),
    label=0,
    normalization=lambda x: x / 255.0,
)
```

Se precisar de uma fonte personalizada (CSV, S3, banco de dados), implemente `IDataSource`:

```python
from simple_duck_ml.dataset_unpacker.i_data_source import IDataSource
from simple_duck_ml.dataset_unpacker.dataset import Dataset

class MeuDataSource(IDataSource):
    def __len__(self) -> int: ...
    def get_sample(self, idx: int) -> Dataset: ...
```

---

## Salvando e carregando modelos

```python
# Salvar
model.save(name="classificador_v1", path="./checkpoints")

# Carregar
from simple_duck_ml import Model
model = Model.load("./checkpoints/classificador_v1")
```

Os pesos são salvos em `.npz` (NumPy) e os metadados em `.toml`.

---

## Redes Neurais Convolucionais — Teoria e Implementação

As seções abaixo explicam o algoritmo de CNN do zero, ligando cada conceito matemático ao arquivo e linha correspondente no framework.

---

### Layer Inicial — Estrutura espacial da entrada

Em uma rede neural **densa**, os nós da camada inicial são um vetor — cada nó recebe um escalar sem noção de posição. Em uma rede **convolucional**, a camada inicial é uma matriz 2D onde cada nó está posicionado em uma coordenada de altura $i$ e largura $j$, preservando a estrutura espacial da entrada:

$$
\text{InputLayer}_{ij}=\begin{bmatrix}
l_{11} & l_{12} & \cdots & l_{1j} \\
l_{21} & l_{22} & \cdots & l_{2j} \\
\vdots  &        & \ddots & \vdots \\
l_{i1} & \cdots &        & l_{ij}
\end{bmatrix}
$$

---

### Formato do input X — Tensores e channels

Para cada coordenada $(i, j)$ da layer inicial existe um vetor chamado **channel**. O input $X$ completo é um tensor tridimensional $(canal \times altura \times largura)$.

- Para **imagens coloridas**: channel = `[R, G, B]` — 3 valores por pixel.
- Para **imagens em escala de cinza**: channel = `[intensidade]` — 1 valor por pixel.
- Para **vídeo**: channel = `[R, G, B, T]` onde $T$ é o instante de tempo.

$$
X_{kij} = \left\lbrace
\begin{bmatrix} x_{111} & \cdots & x_{11j} \\ \vdots & \ddots & \vdots \\ x_{1i1} & \cdots & x_{1ij} \end{bmatrix},\;
\begin{bmatrix} x_{211} & \cdots & x_{21j} \\ \vdots & \ddots & \vdots \\ x_{2i1} & \cdots & x_{2ij} \end{bmatrix},\;
\begin{bmatrix} x_{311} & \cdots & x_{31j} \\ \vdots & \ddots & \vdots \\ x_{3i1} & \cdots & x_{3ij} \end{bmatrix}
\right\rbrace
$$

**No framework:** `ConvolutionalLayer.forward()` aceita tensores `(H, W, C)`. Quando a entrada é 2D (escala de cinza), o canal é adicionado automaticamente — veja [`convolutional_layer.py:70-71`](src/simple_duck_ml/layers/convolutional_layer.py).

---

### Perceptrons de uma layer convolucional — Patches

Em layers densas, cada perceptron recebe **todos** os dados da camada anterior de uma vez. Em layers convolucionais, o perceptron percorre a entrada aos poucos, recortando **patches** (janelas) e processando cada um individualmente.

Um **patch** é uma sub-região do input com as mesmas dimensões do kernel do perceptron. O conjunto de todos os patches gerados para uma entrada é o que a layer convolucional processa.

**No framework:** A extração de patches acontece em `ConvolutionalLayer.__get_patches()` — [`convolutional_layer.py:40-67`](src/simple_duck_ml/layers/convolutional_layer.py). Para cada posição $(i, j)$ do mapa de saída, um patch da entrada é recortado e armazenado:

```python
patches = np.zeros((output_h, output_w, kernel_h, kernel_w, in_channels))
for i, j in np.ndindex(output_h, output_w):
    patches[i, j] = x[i*stride : i*stride+kernel_h, j*stride : j*stride+kernel_w, :]
```

---

### Kernel — O filtro do perceptron

O **kernel** é a janela de captura do perceptron convolucional. Ele define:
- Quantos pixels de altura e largura o perceptron enxerga por vez.
- Quantos canais do input são consumidos.

Formato: $(altura \times largura \times canais)$. Um kernel $(3 \times 3 \times 3)$ gera **27 pesos** e **1 bias**:

$$
\text{Kernel}_{3\times3\times3} =
\begin{bmatrix}
\begin{bmatrix} w_{r11} & w_{r12} & w_{r13} \\ w_{r21} & w_{r22} & w_{r23} \\ w_{r31} & w_{r32} & w_{r33} \end{bmatrix},\;
\begin{bmatrix} w_{g11} & w_{g12} & w_{g13} \\ w_{g21} & w_{g22} & w_{g23} \\ w_{g31} & w_{g32} & w_{g33} \end{bmatrix},\;
\begin{bmatrix} w_{b11} & w_{b12} & w_{b13} \\ w_{b21} & w_{b22} & w_{b23} \\ w_{b31} & w_{b32} & w_{b33} \end{bmatrix}
\end{bmatrix} + b
$$

**No framework:** Os pesos são inicializados com He initialization em [`convolutional_layer.py:26-28`](src/simple_duck_ml/layers/convolutional_layer.py):

```python
limit = np.sqrt(2.0 / np.prod(self.kernel_shape))
self.w = np.random.randn(nodes_num, *self.kernel_shape) * limit  # shape: (N, H, W, C)
self.b = np.zeros((nodes_num, 1))
```

---

### Stride — O passo do kernel

O **stride** controla quantas posições o kernel avança a cada passo.

- `stride=1`: o kernel se move pixel a pixel → mapa de ativação maior, mais detalhado.
- `stride=2`: o kernel pula de 2 em 2 → mapa de ativação menor, mais comprimido.

O tamanho do mapa de ativação resultante é:

$$
\text{output\\_h} = \frac{H_{\text{entrada}} - H_{\text{kernel}}}{\text{stride}} + 1
$$

**No framework:** Calculado em [`convolutional_layer.py:55-56`](src/simple_duck_ml/layers/convolutional_layer.py):

```python
output_h = (input_h - kernel_h) // stride + 1
output_w = (input_w - kernel_w) // stride + 1
```

---

### Forward pass convolucional — Calculando o mapa de ativação

Para cada patch, o perceptron faz o produto elemento a elemento entre o patch e os pesos do kernel, soma tudo, adiciona o bias e passa pela função de ativação. O resultado é sempre **um escalar**:

$$
z_{ij} = \sum_k \sum_{p}\sum_{q} x_{k,i+p,\, j+q} \cdot w_{k,p,q} + b
$$

$$
\text{activationMap}_{ij} = \text{ReLU}(z_{ij})
$$

O agrupamento de todos esses escalares numa matriz 2D é o **mapa de ativação**. Cada perceptron (nó) da layer convolucional produz um mapa de ativação independente.

**No framework:** [`convolutional_layer.py:81-98`](src/simple_duck_ml/layers/convolutional_layer.py). Os patches são achatados em vetores de linha e os pesos em vetores de coluna para que um único `np.dot` compute todos os patches de todos os nós de uma vez:

```python
x_flat = patches.reshape(output_h * output_w, kernel_h * kernel_w * in_channels)
w_flat = self.w.reshape(self.nodes_num, -1)
z = np.dot(w_flat, x_flat.T) + self.b          # shape: (nodes, H_out * W_out)
z = z.reshape(self.nodes_num, output_h, output_w)
self._output = self.activation(z)               # ReLU aplicado elemento a elemento
```

---

### Função de ativação ReLU

ReLU descarta ativações negativas, introduzindo não-linearidade sem custo computacional alto:

$$
\text{ReLU}(z) = \max(0, z)
$$

**No framework:** [`relu_activation.py`](src/simple_duck_ml/activations/relu_activation.py). A implementação usa `clip` para evitar ativações muito grandes (ReLU6):

```python
def __call__(self, x):
    return np.clip(np.maximum(0, x), 0, self.max_value)

def derivative(self, x):
    return (x > 0).astype(np.float64)
```

---

### FlattenLayer — Conectando conv com dense

A saída de uma layer convolucional é um tensor 3D `(nodes, H_out, W_out)`. Antes de entrar numa layer densa, precisa ser achatada num vetor coluna `(features, 1)`.

**No framework:** [`flatten_layer.py:17-42`](src/simple_duck_ml/layers/flatten_layer.py). Guarda o shape original para o backward poder reconstruir o tensor:

```python
def forward(self, x):
    self._input_shape = x.shape
    return x.reshape(-1, 1)

def backward(self, delta):
    return delta.reshape(self._input_shape)  # reconstrói o tensor para a conv anterior
```

---

### Forward pass denso — Layer de classificação

Após o flatten, a layer densa computa:

$$
z = W \cdot x + b \qquad y = \text{Softmax}(z)
$$

O **Softmax** transforma os logits em probabilidades que somam 1 — cada saída representa a confiança do modelo em cada classe.

**No framework:** [`dense_layer.py:41-57`](src/simple_duck_ml/layers/dense_layer.py). Os pesos são inicializados na primeira chamada de `forward`, quando o tamanho de entrada é conhecido:

```python
def forward(self, x):
    if self.w is None:
        self._init_params(x.shape[0])   # He initialization
    z = np.dot(self.w, x) + self.b
    self._x = x                          # guardado para o backward
    self._output = self.activation(z)
    return self._output
```

---

### Função de perda — Cross-Entropy

Mede a diferença entre a distribuição prevista e a distribuição real (one-hot). Quanto menor, mais o modelo acertou:

$$
\mathcal{L} = -\sum_c y_c \cdot \log(\hat{y}_c)
$$

**No framework:** [`cross_entropy_loss.py`](src/simple_duck_ml/loss/cross_entropy_loss.py):

```python
def __call__(self, y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)   # evita log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
```

---

## Backward pass — Aprendendo com os erros

### Gradiente inicial — Derivada da perda

O backward começa calculando o gradiente da função de perda em relação à saída da última layer. Para Cross-Entropy + Softmax, a derivada simplifica para:

$$
\delta = \hat{y} - y_{true}
$$

**No framework:** [`cross_entropy_loss.py:17-18`](src/simple_duck_ml/loss/cross_entropy_loss.py) e chamado em [`model.py:31-33`](src/simple_duck_ml/models/model.py):

```python
def derivative(self, y_pred, y_true):
    return y_pred - y_true
```

---

### Backward denso

Com o delta recebido da camada seguinte, a layer densa calcula:

1. **Delta dos pesos** — quanto cada peso contribuiu para o erro:
$$\nabla W = \delta \cdot x^T$$

2. **Delta do bias**:
$$\nabla b = \delta$$

3. **Delta propagado para trás** — passado à layer anterior:
$$\delta_{anterior} = W^T \cdot \delta$$

**No framework:** [`dense_layer.py:60-78`](src/simple_duck_ml/layers/dense_layer.py):

```python
delta *= self.activation.derivative(self._z)   # aplica derivada da ativação
self._grad_w += np.dot(delta, self._x.T) / batch_size
self._grad_b += np.mean(delta, axis=1, keepdims=True)
return np.dot(self.w.T, delta)                 # propaga para a layer anterior
```

---

### Backward flatten

Apenas reconstrói o tensor com o shape original da saída convolucional:

$$
\delta_{conv} = \text{reshape}(\delta_{flat},\; \text{shape original})
$$

**No framework:** [`flatten_layer.py:45-50`](src/simple_duck_ml/layers/flatten_layer.py).

---

### Backward convolucional

É o passo mais complexo. Para cada posição $(i, j)$ do mapa de ativação:

**1. Aplicar derivada da ativação:**
$$
\delta[i,j] = \delta[i,j] \cdot \text{ReLU}'(\text{output}[i,j])
$$

**2. Gradiente dos pesos** — acumula a contribuição de cada patch:
$$
\nabla W \mathrel{+}= \text{patch}[i,j] \cdot \delta[i,j]
$$

**3. Gradiente do bias** — acumula o delta escalar de cada posição:
$$
\nabla b \mathrel{+}= \delta[i,j]
$$

**4. Delta propagado para trás** — distribui o gradiente de volta ao input:
$$
\delta_{input}[i:i+H_k,\; j:j+W_k] \mathrel{+}= \delta[i,j] \cdot W
$$

**No framework:** [`convolutional_layer.py:100-144`](src/simple_duck_ml/layers/convolutional_layer.py). O acúmulo de $\nabla W$ e $\nabla b$ é vetorizado com `np.dot`; a reconstrução do `grad_x` usa loop sobre patches por ser uma acumulação com sobreposição:

```python
delta *= self.activation.derivative(self._output)

self._grad_w += np.dot(delta_flat, x_flat).reshape(self.w.shape)
self._grad_b += np.sum(delta, axis=(1, 2)).reshape(self._grad_b.shape)

grad_x_flat = np.dot(delta_flat.T, w_flat)
grad_x = np.zeros_like(self._x)
for idx, (i, j) in enumerate(np.ndindex(output_h, output_w)):
    patch_grad = grad_x_flat[idx].reshape(kernel_h, kernel_w, in_channels)
    grad_x[i:i+kernel_h, j:j+kernel_w, :] += patch_grad   # sobreposição intencional
```

---

### Atualização dos pesos — Gradient Descent

Após processar o batch inteiro, os pesos são atualizados com o gradiente médio acumulado:

$$
W \mathrel{-}= \eta \cdot \frac{\nabla W}{\text{batch\\_size}}
\qquad
b \mathrel{-}= \eta \cdot \frac{\nabla b}{\text{batch\\_size}}
$$

**No framework:** `layer.update()` chamado em [`model.py:36-38`](src/simple_duck_ml/models/model.py) após cada mini-batch. Cada layer limpa seus gradientes acumulados ao final do `update()` via `clean_grad()`.

---

### Fluxo completo de treino

```
for cada epoch:
    embaralha os índices das amostras
    for cada mini-batch:
        for cada amostra do batch:
            forward: InputLayer → ConvLayer → FlattenLayer → DenseLayer → Softmax
            calcula perda (CrossEntropy)
            backward: delta ← ∂Loss/∂ŷ → Dense ← Flatten ← Conv
            acumula ∇W e ∇b em cada layer
        update: W -= η * (∇W / batch_size)  para todas as layers
    reporta loss média da epoch
```

**No framework:** [`model.py:45-89`](src/simple_duck_ml/models/model.py).

---

## Referência da API

### `Model`
| Método | Descrição |
|--------|-----------|
| `fit(sources, epochs, batch_size, shuffle, verbose)` | Treina o modelo |
| `forward(x)` | Passa `x` pela rede e retorna a predição |
| `save(name, path, overwrite)` | Salva pesos e metadados em disco |
| `Model.load(path)` | Carrega modelo salvo |

### Layers
| Classe | Parâmetros principais |
|--------|-----------------------|
| `ConvolutionalLayer` | `nodes_num`, `kernel_shape`, `activation`, `stride=1` |
| `FlattenLayer` | — |
| `DenseLayer` | `output_size`, `activation` |

### Ativações
| Classe | Uso típico |
|--------|------------|
| `ReLuActivation` | Layers convolucionais e densas ocultas |
| `SoftmaxActivation` | Última layer (classificação) |

### Perdas
| Classe | Uso típico |
|--------|------------|
| `CrossEntropyLoss` | Classificação multiclasse |
| `MSELoss` | Regressão |

### Fontes de dados
| Classe | Quando usar |
|--------|-------------|
| `InMemoryDataSource(dataset)` | Dataset pequeno já carregado |
| `StreamingDataSource(unpacker, label, normalization)` | Dataset grande, leitura do disco |
| `IDataSource` | Interface para fontes customizadas |

