---
aliases:
- /markdown/2020/07/21/fastbook_ch5
categories:
- ai
date: '2020-07-21'
description: Notes on FastBook.
layout: post
title: FastBook Chapter 5 Thoughts
toc: true

---

# FastBook Chapter 5 Thoughts

After reading the first half of chapter 3 (will read the second half this week) and (mostly) breezing through Chapter 4 (it contained a lot of familiar material), I worked on Chapter 5 this weekend.

## Import concepts:

* Presizing.
* Checking your DataBlock before you begin training
* Train early (get a reasonable MVP) and often (if it's not too expensive).
* Cross-Entropy Loss for the binary case and extending it to multi-class examples.
* Confusion matrix with `ClassificationInterpretation` and looking at the `most_confused` examples.
* Learning Rate Finder
* More particulars on transfer learning, including how to use discriminative learning rates to not lose the solid training of the transferred modeled.

## Some Notes During Reading

> Presizing is a particular way to do image augmentation that is designed to minimize data destruction while maintaining good performance.
The general idea is to apply a composition of the augmentation operations all at once rather than iteratively augment and interpolate. This has savings both in terms of computation and the final quality of the examples.

In the 3s and 7s table there is a column labeled "loss", which for me was a bit confusing. In the first row loss was the predicted output of the "3" class, which happened to be the correct answer. However, loss was just the output of that example, which does not quite make sense because you are looking to minimize loss which conflicts with the goal of maximizing the predicted output for the true class. It looks like this was just an oversight with the naming convention because to compute the loss more things are done and the text that follows makes this clear.

I found it useful to explicitly calculate the loss in the binary example provided.

Activations:

```python
acts[0, :]
```

> tensor([0.6734, 0.2576])

```python
class0_act = acts[0, 0]
class1_act = acts[0, 1]
class0_act
```

> tensor(0.6734)

Computing the exponential of the activations to then get the softmax.

```python
from math import exp
exp0 = exp(class0_act)
exp0
```

> 1.9608552547588787

```python
exp1 = exp(class1_act)
smax0 = exp0 / (exp0 + exp1)
smax0
```

> 0.602468670578454

```python
smax1 = exp1 / (exp0 + exp1)
smax1
```

> 0.39753132942154595

```python
smax0 + smax1
```

> 1.0

```python
from math import log
log(smax0)
```

> -0.5067196140092344

```python
log(smax1)
```

> -0.9224815318387478

```python
-log(smax0)
```

> 0.5067196140092344

And that is the loss for the first example, because the true class was 0. This matches the calculation using the fastai classes, which is always a relief.

## References to Read

* [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
* [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)

## Thoughts on selected Qs

1. Why do we first resize to a large size on the CPU, and then to a smaller size on the GPU?

    You want to create a uniform input size for your data and also apply various transformations to augment it. The presizing method, running augmentation transformations as a single composition rather than iteratively, allows you to have larger/more "rich" inputs to transform before making them a smaller, uniform size that you will train the model with.

2. What are the two ways in which data is most commonly provided, for most deep learning datasets?

    1. A collection of data elements, that have filenames indicating information about them, like their class. (A folder of pictures where each picture has a file name with its ID and class).
    2. Tabular format that can either contain the data in each row (along with the metadata) or point to data in other formats. (A csv file with ID, true class, and a hyper link to the input picture.)

3. Look up the documentation for `L` and try using a few of the new methods is that it adds.

    `L` is a [beefier list class](https://fastcore.fast.ai/foundation#L). How it's different from a regular list:

    * the print function is smarter. It provides the length of the list and truncates the end, which is nice if you've ever crashed a server because you accidentally printed out an obscenely long list.
    * you can access L with a tuple, whereas a normal list will break if you try to access it that way.
    * it has `unique()`, which functions like the same method in Pandas.
    * it has a filter method attribute.

4. Look up the documentation for the Python `pathlib` module and try using a few methods of the `Path` class.

    [Path](https://docs.python.org/3/library/pathlib.html) was introduced to Python in 3.4. It appears to combine a bunch of common things that you typically use `os` with along with the ability to manage file paths without doing string manipulations (as well as reducing the `\` vs `/` mistakes that are frequently made).

    One nice thing that can be done, set `here = Path('.')` and then iterate over the current directory with `for f in here.iterdir(): print(f)`. You can also `.open()` a path object rather than feeding it to `open()` and do `glob` stuff.

5. Give two examples of ways that image transformations can degrade the quality of the data.

    1. Image simply rotating a square 45 degrees to stand it up on one corner. Now the new image that you get (take an old square position cut out of the rotated square position) is missing anything in the corners, so it has to be interpolated. This loses about 17% of the original image, so it's pretty significant!
    2. Brightening an image will move the brighter pixels up to the maximum brightness, so their original brightness cannot be recovered by simply redarkening.

6. What method does fastai provide to view the data in a `DataLoaders`?

    You can use `.show_batch(nrows, ncols)` on the DataLoaders object to get a grid of some of the examples.

7. What method does fastai provide to help you debug a `DataBlock`?

    Using `.summary()` on the DataBlock object gives a verbose attempt to try and create the batch. The output from this, along with errors that come up if it fails can help you notice a problem.

8. Should you hold off on training a model until you have thoroughly cleaned your data?

    No, sometimes life is easy! Also, it's good to get reasonable bench marks as soon as possible. Not only to help game-ify the problem and motivate you to work on it, but also to have a baseline to see if the room for improvement is worth the energy.

9. What are the two pieces that are combined into cross-entropy loss in PyTorch?

    `nn.CrossEntropyLoss` ([see the docs](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss)) applies `nll_loss` after `log_softmax` (which is `log` of `softmax`)

10. What are the two properties of activations that softmax ensures? Why is this important?

    1. You can interpret activations as probabilities.
        * outputs sum to one
        * outputs are non-negative

    2. It forces the model to favor a single class.

    This more relevant behavior that is mentioned that it amplifies small differences, which is useful if you want the network to be somewhat decisive rather than having all outputs close to each other.

11. When might you want your activations to not have these two properties?

    The parenthetical comment in the main text mentions that you may not want the model to pick a class just because it has a slightly larger output. You want the model to be sure about the class, not just relatively sure.

    For the probability property, it might be misleading because it isn't necessarily the actual probability of the example being that class.

12. Why can't we use `torch.where` to create a loss function for datasets where our label can have more than two categories?

    In part, this is a constraint of the where function. Where selects between two outputs based on a condition. It is too difficult to right a nested condition when you have more than two outcomes and selecting the loss requires a bit more work, so this trick becomes way less convenient.

13. What are two good rules of thumb for picking a learning rate from the learning rate finder?

    > * One order of magnitude less than where the minimum loss was achieved (i.e., the minimum divided by 10).
    > * The last point where the loss was clearly decreasing 

14. What two steps does the `fine_tune` method do?

    We go to the [source](https://github.com/fastai/fastai2/blob/master/fastai2/callback/schedule.py#L141):

    ```python
    self.freeze()
    self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    base_lr /= 2
    self.unfreeze()
    self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)
    ```

    Thus it:
        1. Performs a one-cycle fit with the pre-trained layers frozen (their weights do not update).
        2. Performs another one-cycle fit with the pre-trained layers unfrozen at half the learning rate.

15. What are discriminative learning rates?

    The idea here is that you may want weights in certain layers to change at a different rate. In particular, if your first layers come from a pretrained network you may want to do updates to them more slowly than the last layers which are tailored to your particular problem.

16. How is a Python `slice` object interpreted when passed as a learning rate to fastai?

    It acts like `numpy.linspace` where the `num` is implicitly defined as the number of layers.

17. Why is early stopping a poor choice when using 1cycle training?

    For a description of 1cycle training, the [fastai docs](https://docs.fast.ai/callbacks.one_cycle.html) refer to the rate finder paper in the references as well as [this blog post](https://sgugger.github.io/the-1cycle-policy.html). It looks like the basic idea is stopping early does give the training a chance to be finely tuned, because you are likely stopping at a point where the learning rate is still large.

18. What is the difference between `resnet50` and `resnet101`?

    Both `resnet50` and `resnet100` are residual networks, and seem to have been introduced in [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). The basic idea of deep residual networks seems to be "wire a network that is trying to learn the function $\mathcal{H}(x)$ such that it has to learn $\mathcal{H}(x) - x$ instead." The intuition being that, for example, it is easier to learn the zero function than it is to learn the identity function. `resnet-50` looks like it was obtain from the `resnet-34` architecture by replacing certain layer pairs with layer triplets known as bottleneck blocks. `resnet-101` (and `resnet-152`) are just an expansion of this idea, adding 17 more (or 34 more) of these triplet-layer blocks.

19. What does `to_fp16` do?

    > The other downside of deeper architectures is that they take quite a bit longer to train. One technique that can speed things up a lot is mixed-precision training. This refers to using less-precise numbers (half-precision floating point, also called fp16) where possible during training. As we are writing these words in early 2020, nearly all current NVIDIA GPUs support a special feature called tensor cores that can dramatically speed up neural network training, by 2-3x. They also require a lot less GPU memory. To enable this feature in fastai, just add to_fp16() after your Learner creation (you also need to import the module).
