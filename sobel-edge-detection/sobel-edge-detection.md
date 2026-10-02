## What Are Edges?

Edges in images are boundaries where pixel intensity changes sharply. They correspond to:
- Object boundaries
- Surface orientation changes
- Changes in material or lighting
- Depth discontinuities

Detecting edges is fundamental to many computer vision tasks: object detection, segmentation, and feature extraction.

---

## The Gradient Approach

Edges occur where the image intensity changes rapidly. Mathematically, this is where the **gradient** is large:

$$
\nabla I = \left( \frac{\partial I}{\partial x}, \frac{\partial I}{\partial y} \right)
$$

The gradient has two components:
- $\partial I / \partial x$: rate of change in horizontal direction
- $\partial I / \partial y$: rate of change in vertical direction

The gradient magnitude tells us edge strength:

$$
|\nabla I| = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2}
$$

---

## The Sobel Operator

The Sobel operator approximates image gradients using convolution with two 3x3 kernels:

**Horizontal gradient kernel (Kx):**

-1  0  1
-2  0  2
-1  0  1

**Vertical gradient kernel (Ky):**

-1 -2 -1
 0  0  0
 1  2  1

These kernels are designed to:
- Compute finite differences (derivative approximation)
- Apply smoothing to reduce noise (weighted average)

---

## How the Kernels Work

**Kx detects vertical edges:**
- Positive on the right, negative on the left
- Large response when left side is dark and right side is bright
- Zero response for uniform regions

**Ky detects horizontal edges:**
- Positive on the bottom, negative on the top
- Large response when top is dark and bottom is bright

The weights (1, 2, 1) provide smoothing perpendicular to the gradient direction.

---

## Step-by-Step Computation

**Step 1: Pad the image**
- Add 1 pixel of zeros around all edges
- Ensures output has same size as input

**Step 2: Compute horizontal gradient**
- Convolve padded image with Kx
- Result: Gx (horizontal gradient at each pixel)

**Step 3: Compute vertical gradient**
- Convolve padded image with Ky
- Result: Gy (vertical gradient at each pixel)

**Step 4: Compute magnitude**

$$
G = \sqrt{G_x^2 + G_y^2}
$$

This gives edge strength at each pixel.

---

## Numerical Example

**Image patch (3x3) from padded image:**

10  10  10
10  10  50
10  10  50

**Gx computation (at center pixel):**
- Apply Kx: (-1*10) + (0*10) + (1*10) + (-2*10) + (0*10) + (2*50) + (-1*10) + (0*10) + (1*50)
- = -10 + 0 + 10 - 20 + 0 + 100 - 10 + 0 + 50
- = 120

**Gy computation (at center pixel):**
- Apply Ky: (-1*10) + (-2*10) + (-1*10) + (0*10) + (0*10) + (0*50) + (1*10) + (2*10) + (1*50)
- = -10 - 20 - 10 + 0 + 0 + 0 + 10 + 20 + 50
- = 40

**Magnitude:** sqrt(120^2 + 40^2) = sqrt(14400 + 1600) = sqrt(16000) = 126.5

This indicates a strong edge at this position.

---

## Gradient Direction

The gradient direction tells us the edge orientation:

$$
\theta = \arctan\left(\frac{G_y}{G_x}\right)
$$

This angle points perpendicular to the edge (in the direction of steepest ascent).

**For the example above:**
- theta = arctan(40/120) = arctan(0.333) = 18.4 degrees

The edge is mostly vertical (gradient points mostly horizontal).

---

## Properties of Sobel

**Strengths:**
- Simple and fast
- Provides both magnitude and direction
- Built-in smoothing reduces noise sensitivity
- Good for general-purpose edge detection

**Weaknesses:**
- Thick edges (not single-pixel precision)
- Sensitive to noise for small gradients
- Fixed scale (only detects edges at one scale)

---

## Sobel vs. Other Edge Detectors

**Simple finite difference:**
- Kernels: [[-1, 1]] and [[-1], [1]]
- No smoothing, very noise-sensitive
- Sobel adds smoothing

**Prewitt operator:**
- Same structure as Sobel but with weights (1, 1, 1)
- Less smoothing than Sobel
- Slightly sharper but more noise-sensitive

**Scharr operator:**
- Weights: (3, 10, 3) instead of (1, 2, 1)
- Better rotational invariance
- Used when more accuracy is needed

**Canny edge detector:**
- Uses Sobel as a component
- Adds non-maximum suppression and hysteresis
- Produces thin, well-localized edges

---

## Applications

**Preprocessing for other algorithms:**
- Feature detection (corners, lines)
- Object detection
- Image segmentation

**Image analysis:**
- Document scanning (find edges of paper)
- Medical imaging (organ boundaries)
- Industrial inspection (defect detection)

**Artistic effects:**
- Cartoon/sketch effects
- Edge enhancement

---

## Implementation Notes

**Padding:**
- Zero padding is standard
- Replication padding can reduce edge artifacts

**Data types:**
- Gradients can be negative (Gx and Gy)
- Use signed integers or floats for intermediate results
- Magnitude is always non-negative

**Normalization:**
- Raw magnitude can be large (up to ~1443 for 8-bit images)
- Often normalize to [0, 255] for display