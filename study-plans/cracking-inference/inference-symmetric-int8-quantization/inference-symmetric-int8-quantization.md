# Symmetric INT8 Quantization

Neural-network tensors are often stored as floating-point numbers. Quantization represents those values with a smaller set of numbers, which can reduce storage and make supported computations cheaper.

This problem uses one of the simplest schemes: every value in a tensor shares one scale, and each value is represented by a signed 8-bit integer code. The function also reconstructs floating-point values so the approximation can be inspected directly.

## Think of a smaller number line

The input may contain many different floating-point values, but the quantized tensor can use only integer codes from $-127$ through $127$.

The scale tells us how much real value one integer step represents. If the scale is $0.02$, code 10 represents approximately $0.2$, code $-10$ represents $-0.2$, and code zero represents exactly zero.

The integer code and scale must be interpreted together. A code of 10 has no fixed real value without knowing the scale that belongs to the tensor.

Symmetric means positive and negative codes use the same step size around zero. There is no offset or zero point in this exercise.

## Choosing the scale

First find the largest absolute input value:

$$
a=\max_i |x_i|
$$

For a nonzero tensor, choose

$$
s=\frac{a}{127}
$$

This maps the largest magnitude to the edge of the allowed code range. A value $a$ maps to 127, while $-a$ maps to $-127$.

Only one $a$ and one $s$ are calculated for the entire tensor. A small value and a large value therefore use the same quantization step.

## Turning values into integer codes

Each input value is divided by the scale, rounded to the nearest integer, and clipped:

$$
q_i=\operatorname{clip}\left(\operatorname{round}\left(\frac{x_i}{s}\right),-127,127\right)
$$

Division expresses the value in units of the quantization step. Rounding selects the closest available integer code. Clipping guarantees that the code remains representable.

The result is then stored with the INT8 dtype. Rounding alone still produces floating-point values, so the dtype conversion is a required part of the output contract.

## A small numerical example

Take the tensor

$$
[-4,-2,0,2,4]
$$

The absolute maximum is 4, giving

$$
s=\frac{4}{127}\approx0.03150
$$

The unrounded codes are approximately $[-127,-63.5,0,63.5,127]$. After rounding, they become $[-127,-64,0,64,127]$ under the tensor library’s rounding rule.

The middle values cannot be represented exactly with this scale, but their codes are close to the ideal positions. Quantization deliberately trades some precision for a smaller representation.

## Dequantization

To reconstruct a floating-point approximation, multiply each code by the same scale:

$$
\hat{x}_i=q_i s
$$

For the example, code 64 reconstructs to approximately $2.016$. The original value was 2, so the error is about $0.016$.

The dequantized tensor must be calculated from the returned codes and scale. It is not a separate approximation created from the original input.

For values that do not clip, ordinary nearest rounding introduces at most roughly half a scale step of absolute error. A smaller scale gives finer reconstruction, while a larger scale creates wider gaps between representable values.

## Why the range stops at minus 127

An INT8 storage type can hold values from $-128$ through $127$. This exercise deliberately uses only $-127$ through $127$.

Using equal positive and negative magnitudes keeps the mathematical mapping exactly symmetric around zero. Code 127 has a matching code $-127$, and equal-magnitude inputs receive equal-magnitude opposite codes.

Code $-128$ must not appear, even though the dtype can physically store it. The specified quantization range is narrower than the storage dtype’s full range.

## The all-zero tensor

If every input is zero, then $a=0$. The normal formula would produce scale zero, and dividing by it would create invalid values.

The required fallback is:

- scale equals one,
- every code equals zero,
- every reconstructed value equals zero.

Scale one is finite and makes dequantization straightforward because zero multiplied by one remains zero. The numerical value of the fallback scale does not affect the reconstruction as long as the zero codes are preserved, but this problem requires exactly one.

The zero check must happen before division.

## Why a single outlier matters

Suppose most values lie near $0.1$, but one value is 100. The shared scale is determined by 100, so one integer step is about $0.787$. Many values near $0.1$ then round to zero.

This is the limitation of per-tensor quantization. One extreme value controls precision everywhere. The next problem introduces per-channel scales to isolate ranges that differ substantially.

For this problem, however, one global scale is mandatory. Creating several scales would compute a different quantization scheme.

## Shapes, layout, and dtypes

Elementwise operations preserve the input shape, so the code tensor and dequantized tensor have exactly the same shape as the input. The scale contains one floating-point value for the complete tensor.

The input may be non-contiguous. A transposed or sliced tensor can have a different memory stride while still representing the same logical elements. Absolute value, maximum, division, rounding, clipping, and multiplication work on those logical elements without reshaping.

There is no reason to force a contiguous copy or flatten with an operation that assumes contiguous storage.

The codes use INT8. The scale and reconstructed tensor remain floating point and preserve finite values for valid input.

## Cost and memory

Finding the absolute maximum visits every element once. Quantizing and dequantizing also perform constant work per element, so the total time is $O(N)$ for $N$ input values.

The code tensor and reconstructed tensor each require $O(N)$ storage. The scale uses constant space.

This educational function returns both codes and a floating reconstruction. A deployed quantized model may store codes and scale while reconstructing values only where needed, but that storage policy is outside the exercise.

## Common mistakes to avoid

- Using the ordinary maximum instead of the absolute maximum fails when the largest magnitude is negative.
- Dividing by 128 changes the required mapping and leaves the positive endpoint unused.
- Allowing code $-128$ breaks the explicitly symmetric range.
- Casting before rounding truncates values instead of selecting the nearest code.
- Omitting clipping can wrap an out-of-range value during INT8 conversion.
- Returning scale zero for an all-zero tensor creates invalid division and dequantization.
- Calculating dequantization from the original input hides the actual quantization error.
- Flattening a non-contiguous tensor with an incompatible view operation can fail unnecessarily.

The complete method is one shared ruler for the tensor. The absolute maximum sets the ruler’s step size, rounding chooses the nearest mark, clipping protects the endpoints, and multiplication by the same scale reconstructs the represented value.
