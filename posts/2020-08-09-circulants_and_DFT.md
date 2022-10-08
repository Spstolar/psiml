---
aliases:
- /fastpages/2020/08/09/circulants_and_DFT
categories:
- fastpages
date: '2020-08-09'
description: Awesome description
hide: false
image: images/circulant_ex_1.svg
layout: post
metadata_key1: metadata_value1
metadata_key2: metadata_value2
search: false
title: Circulant Matrices and the Discrete Fourier Transform
toc: true

---

# Circulant Matrices and the Discrete Fourier Transform

While reading [this great post](https://towardsdatascience.com/deriving-convolution-from-first-principles-4ff124888028) I saved off the paper that it references: [Discovering transforms: a tutorial on circulant matrices, circular convolution, and the discrete Fourier transform](https://arxiv.org/abs/1805.05533) by Bassam Bamieh. This last week, I dug into the paper and found it super readable and interesting. So this is just my digging into some of the details and making pictures to help things make sense for me.

## Discrete Fourier Transform

The Discrete Fourier Transform (DFT) is a very useful operator that takes a tuple of complex numbers to another tuple of complex numbers. You typically see it defined as an operator on length-n complex vectors
$\mathrm{DFT}: \mathbb{C}^n \to \mathbb{C}^n$ as a rule which sends $a \mapsto \hat{a}$:

$$\hat{a}_k := \sum_{l = 0}^{n-1} a_l e^{-i\frac{2\pi}{n}kl}$$

The issue is that this kind of comes out of nowhere. You might see it in a course without knowing anything about complex analysis or linear algebra where the transform will be defined, perhaps used as an excuse to explain some of the basics of complex analysis, then some properties of the transform will be derived and you'll use it as a tool for say, solving ODEs. There is a [lot of depth](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) to this transformation. Rather than discuss its interpretation and uses, I will give some of the high level details of the Bamieh paper which shows how to arrive at it from naturally exploring circular convolutions.

## Define Vectors

For all the examples we will run, we make two simple vectors:

```python
import numpy as np
a = np.arange(5)
b = np.array([2, -1, 3, 2 ,0])
```

```python
a > array([0, 1, 2, 3, 4])
b > array([ 2, -1,  3,  2,  0])
```

## Circular Convolutions

A [convolution](https://en.wikipedia.org/wiki/Convolution) of two functions $f, g$ is a way to produce a third function that has properties of the two components:
$$ f \star g (t) = \int_{-\infty}^\infty f(\tau) g(t - \tau) d \tau $$

Vector convolutions are similar, and we can think of it as a discrete version of the . When you use the [built in functions from numpy for the convolution of two vectors](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html) you are using this definition:
$$ a \star x [k] = \sum_{l = \infty}^{\infty} a[l] x[k - l] $$

The idea is that you are thinking of infinite-dimensional vectors indexed by the integers. So, when you plug in vectors like `[1, 2, 3]` you are interpreting them as `[..., 0, 0, 0, 1, 2, 3, 0, 0, 0, ...]`.

The default mode of `np.convolve` is 'full', but there are also 'same' and 'valid' modes. Full executes the convolution, but clips the output one index after the results are all zeros:

```python
np.convolve(a, b, 'full')
> array([ 0,  2,  3,  7, 13,  9, 18,  8,  0])
```

```python
np.convolve([1, 1, 1], [1, 1, 1], 'full')
> array([1, 2, 3, 2, 1])
```

Same returns an output that has the same size as the bigger input:

```python
np.convolve(a, b, 'same')
> array([ 3,  7, 13,  9, 18])
```

```python
np.convolve([1, 1, 1], [1, 1, 1], 'same')
> array([2, 3, 2])
```

Valid only returns the output where the signals fully overlapped:

```python
np.convolve(a, b, 'valid')
> array([3])
```

```python
np.convolve([1, 1, 1], [1, 1, 1], 'valid')
> array([3])
```

The [`scipy` implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html) is similar

```python
from scipy import signal
signal.convolve(a, b)
> array([ 0,  2,  3,  7, 13,  9, 18,  8,  0])
```

However, we are interested in the **circular** convolution, where we treat the domain as periodic. Rather than summing over all of the integers, we are thinking of doing arithmetic modulo n. Thus it is useful to view vectors are being circularly arranged:

![test2](../../../../images/circle_vector.svg)

Then the circular convolution of two vectors is defined:
$$ a \star x [k] = \sum_{l = 0}^{n-1} a_{k -l} x_l$$
this is described in more detail in the paper.

To visualize this first we use a drawing to indicate the inner product of two vectors by stacking the circular representation of one on top of the other:

![component-wise multiplication](../../../../images/comp_mult.svg)

We multiply components by matching the top circular vector to the bottom one. Using this, we get the circular convolution of two vectors by

1. reversing the first argument vector so that it goes from 0 to $n-1$ clockwise rather than counter-clockwise and
2. performing a bunch of inner products with each rotation of the "convolving vector" that we flipped (here we use $a$) with the second-argument vector  (here we use $x$ or $b$ in the examples).

![circular convolution](../../../../images/circular_convolution.svg)

Thus coordinate `k` of the circular convolution $a \star x$ is the inner product of x with $a$ reversed and then rotated $k$ times.

We implement this in a straightforward fashion, ignoring the fact that it isn't optimized:

```python
def circular_convolution(a, x):
    n = len(a)
    y = np.zeros(n)
    for k in range(n):
        for l in range(n):
            y[k] += a[k-l] * x[l]
    return y
```

For example, we get a new length-5 vector by convolving $a$ and $b$:

```python
circular_convolution(a, b)
> array([ 9., 20., 11.,  7., 13.])
```

## Circular Convolution is Multiplication with a Circulant Matrix

The circulant matrix of a vector is built like this:

![create circulant](../../../../images/circulant_shift.svg)

There is a [`scipy` method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.circulant.html) for constructing them:

```python
from scipy.linalg import circulant
```

```python
C_a = circulant(a)
```

```python
C_b = circulant(b)
```

```python
C_a
> array([[0, 4, 3, 2, 1],
           [1, 0, 4, 3, 2],
           [2, 1, 0, 4, 3],
           [3, 2, 1, 0, 4],
           [4, 3, 2, 1, 0]])
```

```python
C_b
> array([[ 2,  0,  2,  3, -1],
           [-1,  2,  0,  2,  3],
           [ 3, -1,  2,  0,  2],
           [ 2,  3, -1,  2,  0],
           [ 0,  2,  3, -1,  2]])
```

To see how you can make a circulant matrix (again not optimized):

```python
def right_shift(v):
    n = len(v)
    shifted_v = np.zeros(n)
    for i in range(n):
        shifted_v[i] = v[i-1]
    return shifted_v
```

```python
right_shift(a)
> array([4., 0., 1., 2., 3.])
```

```python
def custom_circulant(v):
    n = len(v)
    C_v = np.zeros((n, n))
    C_v[:, 0] = v  # first column is just the vector

    # next columns are each right/down-shifted
    for i in range(1, n):
        C_v[:, i] = right_shift(C_v[:, i-1])

    return C_v
```

```python
custom_circulant(a)
>   array([[0., 4., 3., 2., 1.],
           [1., 0., 4., 3., 2.],
           [2., 1., 0., 4., 3.],
           [3., 2., 1., 0., 4.],
           [4., 3., 2., 1., 0.]])
```

An interesting fact is that circulant matrices commute:

```python
C_a @ C_b == C_b @ C_a
>   array([[ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]])
```

```python
C_a @ C_b
>   array([[ 9, 13,  7, 11, 20],
           [20,  9, 13,  7, 11],
           [11, 20,  9, 13,  7],
           [ 7, 11, 20,  9, 13],
           [13,  7, 11, 20,  9]])
```

But now look, if $C_a$ is the circulant matrix built from $a$, $$a \star x = C_a x$$

```python
C_a @ b == circular_convolution(a, b)
> array([ True,  True,  True,  True,  True])
```

Circular convolution is just multiplying by the circulant matrix and moreover (see the paper)
$$ a \star x = C_a x = C_x a = a \star x$$

## How DFT Is Related

Now let's use the numpy implementation of the discrete fourier transform:

$\hat{a} = \mathrm{DFT}(a)$

```python
a_hat = np.fft.fft(a)
```

$\hat{b} = \mathrm{DFT}(b)$

```python
b_hat = np.fft.fft(b)
```

```python
a_hat
> array([10. +0.j, -2.5+3.4409548j , -2.5+0.81229924j, -2.5-0.81229924j, -2.5-3.4409548j])
```

```python
b_hat
> array([ 6.+0.j, -2.35410197+0.36327126j, 4.35410197+1.53884177j,  4.35410197-1.53884177j, -2.35410197-0.36327126j])
```

Now take the component-wise product of these DFT'd vectors $\hat{a}$ and $\hat{b}$:

```python
hat_prod = a_hat * b_hat
```

```python
hat_prod
> array([ 60.+0.j, 4.63525492-9.00853662j, -12.13525492-0.3102707j , -12.13525492+0.3102707j , 4.63525492+9.00853662j])

```
Now let's undo the DFT using the Inverse Discrete Fourier Transform:

```python
np.fft.ifft(hat_prod)
> array([ 9.+0.j, 20.+0.j, 11.+0.j,  7.+0.j, 13.+0.j])
```
So, $\mathrm{DFT}^{-1}(\hat{a} * \hat{b})$ was `[9, 20, 11, 7, 13]` in our example, all real with no imaginary components.

But wait a second:

```python
circular_convolution(a, b)
> array([ 9., 20., 11.,  7., 13.])
```

Cool!
$$\mathrm{DFT}^{-1}(\hat{a} * \hat{b}) = a \star b$$

## How Does This Work

Hopefully this notebook will interest you enough to dive into the source paper to find out why this "trick" works, but here are some of the details with examples.

First iteresting fact: the DFT of a vector gives the same result as the eigenvalues of the circulant matrix for that vector.

```python
np.fft.fft(a)
> array([10. +0.j        , -2.5+3.4409548j , -2.5+0.81229924j,
           -2.5-0.81229924j, -2.5-3.4409548j ])
```

```python
np.linalg.eigvals(C_a)
> array([10. +0.j        , -2.5+3.4409548j , -2.5-3.4409548j ,
           -2.5+0.81229924j, -2.5-0.81229924j])
```

```python
custom_circulant(a)
>   array([[0., 4., 3., 2., 1.],
           [1., 0., 4., 3., 2.],
           [2., 1., 0., 4., 3.],
           [3., 2., 1., 0., 4.],
           [4., 3., 2., 1., 0.]])
```

Second interesting fact: the circulant matrices commute with the operator that rotates a circular vector. This is because this operator, the step matrix $S$, is itself a circulant matrix.

```python
step_matrix = circulant([0, 1, 0, 0, 0])
```

```python
step_matrix
>   array([[0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0]])
```

So it makes us step right/clockwise:

```python
a
> array([0, 1, 2, 3, 4])
```

```python
step_matrix @ a
> array([4, 0, 1, 2, 3])
```

The adjoint of the step matrix $S^*$ is just the step in the other direction.

```python
step_matrix_adjoint = step_matrix.T
```

```python
step_matrix_adjoint
>   array([[0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0]])
```

```python
step_matrix_adjoint @ a
> array([1, 2, 3, 4, 0])
```

Interesting fact 3: the step matrix is a fundamental example of a circulant matrix. If you look at the eigenvalues and eigenvectors of its adjoint, you recover the DFT as well as connection between the periodic domain and modular arithmetic used in the definition of the circular convolution.

The eigenvalues of $S^*$:

```python
step_eigs = np.linalg.eigvals(step_matrix_adjoint)
```

```python
step_eigs
> array([-0.80901699+0.58778525j, -0.80901699-0.58778525j,
            0.30901699+0.95105652j,  0.30901699-0.95105652j,
            1.        +0.j        ])
```

It may be hard to see a pattern here, so let's plot them on the complex plane.

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
```

```python
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(-1.1, 1.1)
circle = Circle((0, 0), 1, facecolor='none',
                edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
ax.scatter(np.real(step_eigs), np.imag(step_eigs), alpha=1)
```

![png](../../../../images/output_91_1.png)

These are the roots of unity. In this case, because we were looking at $n = 5$, they are the fifth-roots of unity: $\exp \left(\frac{i 2 \pi k}{5} \right)$ for $k = 0, 1, 2, 3, 4$.

You can build a matrix with nice choices of eigenvectors of $S^*$:

$$ \begin{pmatrix}
1 & 1 & 1 & \dots & 1 \\
1 & \rho & \rho^2 & \dots & \rho^{n-1} \\
1 & \rho^{2} & \rho^4 & \dots & \rho^{2(n-1)} \\
\vdots & \vdots & \vdots & & \vdots \\
1 & \rho^{n-1} & \rho^{2(n-1)} & \dots & \rho^{(n-1)(n-1)}
\end{pmatrix} $$

where $\rho$ is the first root of unity. For example $\exp \left(\frac{i 2 \pi}{5} \right)$ for $n = 5$, the top root in that picture.

First we make a way to grab the $n$-th roots:

```python
def roots_of_unity(n):
    # returns solutions to: x^n - 1 = 0
    coeffs = np.zeros(n + 1)
    coeffs[0] = 1  # x^n
    coeffs[-1] = -1  # - 1
    return np.roots(coeffs)
```

Let's verify this by checking the $n$-th power of each of the roots:

```python
[r**2 for r in roots_of_unity(2)]
> [1.0, 1.0]
```

```python
[r**3 for r in roots_of_unity(3)]
> [(1.0000000000000007-3.885780586188048e-16j),
     (1.0000000000000007+3.885780586188048e-16j),
     (0.9999999999999993+0j)]
```

For our example we want:

```python
fifth_roots = roots_of_unity(5)
fifth_roots
> array([-0.80901699+0.58778525j,
         -0.80901699-0.58778525j,
          0.30901699+0.95105652j,
          0.30901699-0.95105652j,
          1.        +0.j        ])
```

It can be helpful to view the real and complex parts separately especially when there are more than 3 significant digits. You can also set the precision displayed:

```python
np.set_printoptions(precision=2)
```

```python
np.real(fifth_roots)
> array([-0.81, -0.81,  0.31,  0.31,  1.  ])
```

```python
np.imag(fifth_roots)
> array([ 0.59, -0.59,  0.95, -0.95,  0.  ])
```

Now let's plot them to make sure they look right:

```python
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(-1.1, 1.1)
circle = Circle((0, 0), 1, facecolor='none',
                edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
ax.scatter(np.real(fifth_roots), np.imag(fifth_roots), alpha=1)
```

![png](../../../../images/output_109_1.png)

So, the way it supplied roots was sort of "out of order". We the one on the top which was at index 2:

```python
first_root = fifth_roots[2]
```

```python
first_root
> (0.30901699437494734+0.9510565162951536j)
```

```python
first_root ** 5
> (1+4.440892098500626e-16j)
```

When checking equality of floats you can run into precision issues, so it's nice to use `np.isclose()`.

```python
np.isclose(first_root ** 5, 1)
> True
```

So it is indeed a fifth root.

We want to show a relationship between the matrix of eigenvectors for the step matrix and DFT, but the default eigenvectors are not in quite the shape we want:

```python
W = np.linalg.eig(step_matrix_adjoint)[1]
```

```python
W
>   array([[ 0.14-0.43j,  0.14+0.43j, -0.36-0.26j, -0.36+0.26j,  0.45+0.j  ],
           [ 0.14+0.43j,  0.14-0.43j,  0.14-0.43j,  0.14+0.43j,  0.45+0.j  ],
           [-0.36-0.26j, -0.36+0.26j,  0.45+0.j  ,  0.45-0.j  ,  0.45+0.j  ],
           [ 0.45+0.j  ,  0.45-0.j  ,  0.14+0.43j,  0.14-0.43j,  0.45+0.j  ],
           [-0.36+0.26j, -0.36-0.26j, -0.36+0.26j, -0.36-0.26j,  0.45+0.j  ]])
```

Which is why we need to do this work to get our desired form

 $$ \begin{pmatrix}
1 & 1 & 1 & \dots & 1 \\
1 & \rho & \rho^2 & \dots & \rho^{n-1} \\
1 & \rho^{2} & \rho^4 & \dots & \rho^{2(n-1)} \\
\vdots & \vdots & \vdots & & \vdots \\
1 & \rho^{n-1} & \rho^{2(n-1)} & \dots & \rho^{(n-1)(n-1)}
\end{pmatrix} $$

So the powers (mod n) are:

```python
n = 5
root_power = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        power = j * i % n
        root_power[i, j] = power
```

```python
root_power
>   array([[0., 0., 0., 0., 0.],
           [0., 1., 2., 3., 4.],
           [0., 2., 4., 1., 3.],
           [0., 3., 1., 4., 2.],
           [0., 4., 3., 2., 1.]])
```

For 5 this is a bit boring because it's prime, but for 4:

```python
n = 4
root_power = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        power = j * i % n
        root_power[i, j] = power

root_power
> array([[0., 0., 0., 0.],
           [0., 1., 2., 3.],
           [0., 2., 0., 2.],
           [0., 3., 2., 1.]])
```

Now we use the first root of unity and raise it to the desired powers to create the matrix of eigenvectors in the nice form:

```python
# might have to do this to get around an error where things are squashed to real parts
W = np.zeros((step_matrix_adjoint.shape), dtype = "complex_")
```

```python
n = 5
for i in range(n):
    for j in range(n):
        power = i * j % n
        W[i, j] = first_root ** power
```

```python
W
> array([[ 1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ,  1.  +0.j  ],
           [ 1.  +0.j  ,  0.31+0.95j, -0.81+0.59j, -0.81-0.59j,  0.31-0.95j],
           [ 1.  +0.j  , -0.81+0.59j,  0.31-0.95j,  0.31+0.95j, -0.81-0.59j],
           [ 1.  +0.j  , -0.81-0.59j,  0.31+0.95j,  0.31-0.95j, -0.81+0.59j],
           [ 1.  +0.j  ,  0.31-0.95j, -0.81-0.59j, -0.81+0.59j,  0.31+0.95j]])
```

Remembering the first root, we see this looks right.

```python
first_root
> (0.30901699437494734+0.9510565162951536j)
```

You can compute the adjoint (transpose and complex conjugate) with:

```python
W_adjoint = W.T.conj()
```

The next interesting fact to think about is (5.1):
$$ C_a W = W \mathrm{diag}(\hat{a}) $$

Create the a diagonal matrix with the entries of $\hat{a}$

```python
diag_a_hat = np.diag(a_hat)
```

```python
np.isclose(C_a @ W, W @ diag_a_hat)
>   array([[ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]])
```

## Summary

What this is saying is that $W$ also provided eigenvectors for $C_a$ and they have eigenvalues from $\mathrm{DFT}(a) = \hat{a}$. This nice matrix not only diagonalizes $S^*$ but it also diagonalizes *all circulant matrices simultaneously*.

This means we can compute DFTs of length $n$ just by knowing the n-th root of unity! (5.2) re-written:
$$\mathrm{Diag}(\hat{a}) = W^{-1} C_a W = \frac 1n W^* C_a W$$

But the point of (5.2) is circular convolution of x with a is just:

1. Take the DFT of x, using multiplication with the adjoint of a nice matrix of eigenvectors of the simple left-shift matrix ($S^*$): 

    $W^* x$

2. Entry-wise multiplication with this result by the DFT of $a$ ($\hat{a}$): 

    $\mathrm{Diag}(\hat{a}) W^* x$

3. Multiply this result by the inverse of the nice adjoint: 

    $\frac 1n W \mathrm{Diag}(\hat{a}) W^* x$

This is cool because not only do we see that DFT naturally comes from looking at shift-invariant transformations but we also have a way of understanding circular convolutions:

* $a \star x = x \star a = C_a x = C_x a = \mathrm{DFT}^{-1} ( \hat{x} \circ \hat{a})$

This last one is especially interesting, convolution is the time-domain equivalent of simple component-wise multiplication in the frequency domain.
