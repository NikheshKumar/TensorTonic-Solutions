<span style="font-size: 14px;">A linear (affine) layer is the core building block of most neural networks. Implementing it from scratch clarifies how parameters are stored and how the forward pass is computed.</span>

## <span style="font-size: 14px;">Mathematical definition</span>

* <span style="font-size: 14px;">For input</span> $x \in \mathbb{R}^{B \times d_{\mathrm{in}}}$ <span style="font-size: 14px;">(batch</span> $B$<span style="font-size: 14px;">, in_features</span> $d_{\mathrm{in}}$<span style="font-size: 14px;">), the layer has weight</span> $W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ <span style="font-size: 14px;">and bias</span> $b \in \mathbb{R}^{d_{\mathrm{out}}}$<span style="font-size: 14px;">. The output is</span>

$$
y = x W^\top + b \in \mathbb{R}^{B \times d_{\mathrm{out}}}.
$$

* <span style="font-size: 14px;">So each row of</span> $x$ <span style="font-size: 14px;">is multiplied by</span> $W^\top$ <span style="font-size: 14px;">and the bias is added. In PyTorch,</span> <code>nn.Linear</code> <span style="font-size: 14px;">stores</span> $W$ <span style="font-size: 14px;">with shape (out_features, in_features) so that</span> <code>x @ weight.T + bias</code> <span style="font-size: 14px;">gives the correct shape.</span>

## <span style="font-size: 14px;">Parameters in nn.Module</span>

* <span style="font-size: 14px;">Parameters must be</span> <code>nn.Parameter</code> <span style="font-size: 14px;">so they are registered with the module and appear in</span> <code>.parameters()</code> <span style="font-size: 14px;">and</span> <code>state_dict()</code><span style="font-size: 14px;">. Weight can be initialized with</span> <code>torch.nn.init.kaiming_uniform_</code> <span style="font-size: 14px;">(or a similar scheme to</span> <code>nn.Linear</code><span style="font-size: 14px;">); bias is often initialized from a uniform range depending on fan-in.</span>
* <span style="font-size: 14px;">Using the same parameter names weight and bias and the same shapes as</span> <code>nn.Linear</code> <span style="font-size: 14px;">allows loading a pretrained</span> <code>nn.Linear</code> <span style="font-size: 14px;">state into your layer.</span>

## <span style="font-size: 14px;">Forward implementation</span>

* <span style="font-size: 14px;">Compute</span> <code>x @ self.weight.T + self.bias</code><span style="font-size: 14px;">. For batched</span> $x$<span style="font-size: 14px;">, the</span> <code>@</code> <span style="font-size: 14px;">operator handles the batch dimension. This is differentiable so autograd will compute gradients for weight and bias in</span> <code>.backward()</code><span style="font-size: 14px;">.</span>
