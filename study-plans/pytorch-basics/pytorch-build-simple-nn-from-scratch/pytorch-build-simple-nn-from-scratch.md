# <span style="font-size: 20px;">Building a Simple Neural Network from Scratch</span>

A minimal feedforward neural network is a stack of linear (affine) layers and nonlinear activations. This problem uses one hidden layer with ReLU, which is the simplest architecture capable of learning nonlinear functions.

## Linear Layers

A linear layer computes the affine transformation $y = xW^\top + b$, where $x$ has shape (batch, in_features), $W$ has shape (out_features, in_features), and $b$ is a bias vector of length out_features. In PyTorch, `nn.Linear(in_features, out_features)` creates such a layer and stores $W$ and $b$ as learnable parameters.

The key insight is that without a nonlinearity between linear layers, the entire network collapses to a single linear transformation. If you stack two linear layers $y = (xW_1^\top + b_1)W_2^\top + b_2$, this simplifies to $y = xW_{\text{eff}}^\top + b_{\text{eff}}$ where $W_{\text{eff}} = W_2 W_1$ and $b_{\text{eff}} = b_1 W_2^\top + b_2$. No matter how many linear layers you chain together, the result is equivalent to a single linear map. This is why activations between layers are essential.

## ReLU Activation

ReLU (Rectified Linear Unit) applies $\text{ReLU}(x) = \max(0, x)$ element-wise. It is the default choice for hidden layers in modern networks for several reasons. First, it is computationally cheap: just a comparison and a selection. Second, for any input $x > 0$ the gradient is exactly 1, so gradients flow through without attenuation during backpropagation. This avoids the vanishing gradient problem that plagued earlier activations like sigmoid and tanh.

Placing ReLU between the two linear layers gives the network the ability to represent nonlinear decision boundaries. The universal approximation theorem guarantees that a single hidden layer with a nonlinear activation can approximate any continuous function on a compact set, given enough hidden units.

## Two-Layer Architecture

With one hidden layer of size $h$, the forward pass proceeds in three steps:

1. **Pre-activation**: Compute $a_1 = xW_1^\top + b_1$, which has shape (batch $\times$ $h$). This is a linear projection from the input space to the hidden space.

2. **Activation**: Apply the nonlinearity $z_1 = \text{ReLU}(a_1)$, which has the same shape (batch $\times$ $h$). This zeros out negative values, creating a piecewise-linear representation. Each hidden unit acts as a "feature detector" that is either active or inactive for a given input.

3. **Output**: Compute $y = z_1 W_2^\top + b_2$, which has shape (batch $\times$ out_features). This is a linear combination of the activated hidden features, producing the final prediction.

The first linear layer plus ReLU is commonly called the "hidden layer," while the second linear layer is the "output layer." For classification tasks, the output is typically passed through a softmax (or handled by `nn.CrossEntropyLoss`, which combines softmax and log-loss internally).

## Implementing with nn.Module

To build this network in PyTorch, you subclass `nn.Module`. In the constructor (`__init__`), you create the three components and assign them as attributes of `self`: `self.linear1 = nn.Linear(in_features, hidden_size)`, `self.relu = nn.ReLU()`, and `self.linear2 = nn.Linear(hidden_size, out_features)`. Assigning them to `self` is not just a convention - it is what registers these as submodules. PyTorch's `nn.Module` machinery inspects all attributes, and any that are themselves `nn.Module` instances get their parameters tracked automatically. This means `model.parameters()` will yield the weights and biases of both linear layers, and `model.state_dict()` will include them for saving and loading.

In the `forward(self, x)` method, you call the layers in sequence: pass `x` through `self.linear1`, then through `self.relu`, then through `self.linear2`, and return the result. PyTorch records each operation in its autograd computational graph, so calling `loss.backward()` later will compute gradients for all parameters.

## Shape Flow

Understanding how tensor shapes transform through the network is critical for debugging. The input $x$ has shape (batch, in_features). The first linear layer projects each sample from $\mathbb{R}^{\text{in\_features}}$ to $\mathbb{R}^{h}$, producing shape (batch, $h$). ReLU operates element-wise and preserves this shape. The second linear layer then maps from $\mathbb{R}^{h}$ to $\mathbb{R}^{\text{out\_features}}$, giving the final output shape (batch, out_features).

The total number of learnable parameters is $(d_{\text{in}} \times h + h) + (h \times d_{\text{out}} + d_{\text{out}})$, where the first term accounts for $W_1$ and $b_1$ and the second for $W_2$ and $b_2$.
