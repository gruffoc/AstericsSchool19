import numpy as np
from typing import (Dict, Tuple, Callable, Sequence, Iterator, NamedTuple)
from numpy import ndarray as Tensor
import matplotlib.pyplot as plt
# un tensore, e` un tensore, se trasforma come un tensore.
Func = Callable[[Tensor], Tensor]


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

# La scelta della funzione di loss da usare e` molto soggettiva al problema
# che si deve affrontare, pero` e` necessario che questa sia convessa,
# altrimenti rischi di venire trappato in qualche minimo locale da cui
# difficilmente ci si riesce a togliere.


class MeanSquareError(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted-actual)


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    """
    Inputs are of size (batch_size, input_size)
    Outputs are of size (batch_size, output_size)
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # Inherit from base class Layer
        super().__init__()  # capire bene
        # Initialize the weights and bias with random values
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (batch_size, input_size)
        W shape is (input_size, output_size)
        b shape is (output_size)
        """
        self.inputs = inputs
        # Compute here the feed forward pass\

        W = self.params['w']
        b = self.params['b']

        return (inputs @ W) + b  # W^T . x + b

    def backward(self, grad: Tensor) -> Tensor:
        """
        grad shape is (batch_size, output_size)
        return shape is (batch_size, input_size)
        gradW shape is the same as W shape
        gradb shape is the same as b shape
        """

        W = self.params['w']
        b = self.params['b']

        # Compute here the gradient parameters for the layer
        self.grads["w"] = self.inputs.T @ grad  # La derivata parziale lungo w del loss al layer i
        self.grads["b"] = np.sum(grad, axis=0)  # La derivata parziale lungo b del loss al layer i
        # Compute here the feed backward pass
        return grad @ self.params['w'].T


class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """

    def __init__(self, f: Func, f_prime: Func) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)  # qui applico la funzione di att al mio dataset

    def backward(self, grad: Tensor) -> Tensor:
        """
        if z = f(x) and y = g(f(x))
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad  # f'(x) * g'(z)


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    return 1-(np.tanh(x)**2)


def sigmoid(x: Tensor) -> Tensor:
    return 1/(1+np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:
    # Write here the derivative of the sigmoid
    return sigmoid(x) * (1 - sigmoid(x))


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """
        The forward pass takes the layers in order
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        """
        The backward pass is the other way around
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for params, grad in net.params_and_grads():
            # Write here the parameters update
            params -= self.lr * grad


# for every epoch - had better to shuffle the data.
Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)


def train(net: NeuralNet, inputs: Tensor, targets: Tensor,
          mse_loss: Loss = MeanSquareError(),
          optimizer: Optimizer = SGD(),
          iterator: DataIterator = BatchIterator(),
          num_epochs: int = 5000) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            # Write here the various steps (in order) needed
            # at each epoch
            X = batch.inputs
            y_true = batch.targets
            # Compute the predicitons of the current network

            y_predicted = net.forward(X)

            # compute the loss
            epoch_loss += mse_loss.loss(y_predicted, y_true)
            # Compute the gradient of the loss
            grad = mse_loss.grad(y_predicted, y_true)
            # Backpropagate the gradients
            net.backward(grad)
            # Update the network
            optimizer.step(net)

        # Print status every 1000 iterations
        if epoch % 4000 == 0:
            print(epoch, epoch_loss)



def prova(x, y):
    return x, y, np.exp((- x**2. - y**2.))

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# X = np.array([[]])
# y = np.array([[]])
#
# xx, yy, res = prova(np.random.uniform(), np.random.uniform())
# ax = [xx, yy]
#
# X = np.append(X ,ax)
# y = np.append(y, res)
#
#
# for i in range(1, 1000):
#     xx, yy, res = prova(np.random.uniform(), np.random.uniform())
#     ax = [xx, yy]
#     X = np.vstack([X, ax])
#     y = np.append(y, res)
#
# y = np.vstack(y)

def print_xor_results(inputs: Tensor, targets: Tensor, predictions: Tensor) -> None:
    print('\nX => y => y_pred')
    for x, y, z in zip(inputs, targets, predictions):
        print(f'{x} => {y} => {z}')


def train_xor(net: Optimizer, inputs: Tensor, targets: Tensor, epochs: int = 4000):
    train(net, inputs, targets, num_epochs=epochs)
    predictions = net.forward(inputs)
    print_xor_results(inputs, targets, predictions)


def plot_decision_contours(network, keras=False, bounds=[0, 1, 0, 1], **kwargs):
    # Create an array of points to plot the decision regions
    x_min, x_max, y_min, y_max = bounds
    rows, cols = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    X_grid = np.c_[rows.ravel(), cols.ravel()]
    # Apply the decision function on the two vectors
    if keras:
        values = network.predict(X_grid)
    else:
        values = network.forward(X_grid)
    # Reshape the array to recover the squared shape
    values = values.reshape(rows.shape)

    plt.figure(figsize=(5, 5))
    # Plot decision region
    plt.pcolormesh(rows, cols, values > 0.5,
                   cmap='Paired')
    plt.grid(False)
    # Plot decision boundaries
    plt.contour(rows, cols, values,
                levels=[.25, .5, .75],
                colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


#net1 = NeuralNet([Linear(input_size=2, output_size=1), ])
#train_xor(net1, X, y)
#plot_decision_contours(net1)

# net2 = NeuralNet([
#     Linear(input_size=2, output_size=4),
#     # Qui se vuoi ci infili dentri una funzione di attivazione!!!!
#     Linear(input_size=4, output_size=1),
#     ])
# train_xor(net2, X, y)
# plot_decision_contours(net2)

net2 = NeuralNet([
    Linear(input_size=2, output_size=4),
    Tanh(),
    # Qui se vuoi ci infili dentri una funzione di attivazione!!!!
    # Linear(input_size=4, output_size=4),
    # Tanh(),
    # Linear(input_size=4, output_size=4),
    # Tanh(),
    # Linear(input_size=4, output_size=4),
    # Tanh(),
    Linear(input_size=4, output_size=1),
    ])
train_xor(net2, X, y)
plot_decision_contours(net2)
