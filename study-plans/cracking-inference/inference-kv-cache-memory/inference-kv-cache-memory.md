# Calculating KV-Cache Memory

A KV cache saves computation by storing attention keys and values from earlier tokens. The tradeoff is memory: every active sequence keeps cached information for every processed token and every transformer layer.

This problem calculates that storage for four attention designs. The arithmetic is straightforward once we identify exactly what each design stores per token.

The final answer is measured in bytes. It depends on the batch size, cached sequence length, number of layers, number and width of cached heads, and bytes used by each stored number.

## Start with the common factors

Every batch item has a cache. Every cached token contributes an entry at every layer. Every stored scalar occupies a fixed number of bytes.

These shared factors can be grouped as

$$
C=B\times S\times L\times b
$$

where $B$ is batch size, $S$ is sequence length, $L$ is layer count, and $b$ is bytes per element.

The remaining question is the number of cached scalar values per token and layer. That part changes across MHA, MQA, GQA, and MLA.

## Multi-head attention

Multi-head attention, or MHA, gives every query head its own key head and value head. If there are $H_q$ query heads and each head has width $D$, one token stores

$$
H_qD\text{ key values}+H_qD\text{ value values}
$$

The factor of two represents the separate key and value tensors. Total MHA cache memory is

$$
M_{\text{MHA}}=B\times S\times L\times 2H_qD\times b
$$

Doubling the cached sequence length doubles this memory. Doubling the batch size or layer count does the same.

## Multi-query attention

Multi-query attention, or MQA, lets all query heads share one key head and one value head. The number of query heads no longer multiplies cache storage.

The formula becomes

$$
M_{\text{MQA}}=B\times S\times L\times 2D\times b
$$

There is still a factor of two because the cache contains one key vector and one value vector. MQA reduces the head count in the cache, not the existence of the two cached tensors.

For the same model dimensions, the ratio between MHA and MQA cache memory is $H_q$. Eight MHA heads use eight times the conventional KV storage of one shared MQA head.

## Grouped-query attention

Grouped-query attention, or GQA, lies between MHA and MQA. Query heads are divided into groups, and each group shares one key head and one value head.

If $H_{kv}$ is the supplied number of GQA key-value heads, then

$$
M_{\text{GQA}}=B\times S\times L\times 2H_{kv}D\times b
$$

The supplied KV-head count must be used directly. Substituting the query-head count would accidentally calculate MHA memory.

Two boundary cases are useful checks:

- when $H_{kv}=H_q$, GQA memory equals MHA memory,
- when $H_{kv}=1$, GQA memory equals MQA memory.

These equalities confirm that the three conventional formulas use the same storage convention.

## Multi-head latent attention

In the MLA convention defined by this problem, the cache does not store a full key vector and a full value vector for every KV head. It stores one shared compressed latent of width $D_c$ and one separate rotary-key component of width $D_r$ per token and layer.

Its memory is

$$
M_{\text{MLA}}=B\times S\times L\times(D_c+D_r)\times b
$$

There is no factor of two in this formula. The latent is a single shared cached representation under the stated convention, and the rotary-key component is added once.

There is also no query-head or KV-head multiplier. Adding either would discard the compression assumption that this calculation is meant to represent.

MLA implementations can differ in their exact cached components. This exercise defines the convention explicitly, so the calculation must follow the supplied latent and rotary widths rather than importing another implementation’s layout.

## A complete numerical example

Suppose the batch size is 1, the sequence length is 128, and the model has 2 layers. MHA has 8 query heads, GQA has 2 KV heads, and the head width is 64. The MLA latent and rotary-key widths are 32 and 16, and each stored element uses 2 bytes.

The common factor is

$$
C=1\times128\times2\times2=512
$$

MHA stores $2\times8\times64=1024$ scalars per token and layer, giving $512\times1024=524{,}288$ bytes.

MQA stores $2\times1\times64=128$ scalars per token and layer, giving $65{,}536$ bytes.

GQA stores $2\times2\times64=256$ scalars per token and layer, giving $131{,}072$ bytes.

MLA stores $32+16=48$ scalars per token and layer, giving $24{,}576$ bytes.

This example shows why the cache layout matters. All four designs process the same number of tokens, yet their stored widths differ substantially.

## Element size changes storage directly

The bytes-per-element input converts scalar counts into byte counts. Common values in this problem are:

- 1 byte for FP8,
- 2 bytes for FP16 or BF16,
- 4 bytes for FP32.

Changing from two-byte storage to one-byte storage halves the calculated cache size. This formula counts only storage and does not claim that two formats have identical numerical behavior or hardware support.

Do not convert to kilobytes or gigabytes inside the result. The function must return exact byte counts in the order MHA, MQA, GQA, MLA.

## Why 64-bit results are required

Long contexts, many layers, and large batches create products that can exceed the largest signed 32-bit integer, which is a little over two billion.

For example, multiplying a 200,000-token context by dozens of layers and many heads can produce tens or hundreds of billions of bytes. An overflowing 32-bit integer may wrap to a negative or otherwise incorrect value.

Use exact integer arithmetic for every intermediate product and return 64-bit integers. Floating-point arithmetic is unnecessary because all dimensions and byte sizes are integers.

## Complexity

The function evaluates four closed-form expressions. Its running time and auxiliary memory are both $O(1)$ with respect to the model dimensions because it does not create a cache or iterate over tokens.

The values being estimated still grow linearly with batch size, sequence length, and layer count. Constant-time calculation should not be confused with constant cache size.

## Common mistakes to avoid

- Forgetting the factor of two for conventional key and value tensors halves MHA, MQA, and GQA estimates.
- Multiplying MQA by the query-head count removes its shared-head memory advantage.
- Using query heads instead of the supplied GQA KV heads calculates the wrong architecture.
- Adding a factor of two or a head multiplier to MLA violates this problem’s cache convention.
- Omitting the MLA rotary-key width undercounts its storage.
- Returning values in elements rather than bytes ignores the element-size input.
- Using 32-bit arithmetic can overflow on valid long-context configurations.
- Returning approximate megabytes loses the exact integer result required by the function.

The reliable method is to separate common scale from per-token layout. Batch, sequence, layers, and element size tell us how many times storage repeats; the attention design tells us how many scalars are stored each time.
