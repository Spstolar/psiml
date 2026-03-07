---
aliases:
- /markdown/2020/07/11/fastai_book_ch2
categories:
- ai
date: '2020-07-11'
description: Simple construction of image classifiers.
layout: post
title: FastAI Book Chapter 2
toc: true

---

# FastAI Book Chapter 2
I went through the second chapter of the book today, which is why this blog even exists.
Highly useful; looking forward to working on some music projects to really learn the material.

## Things I Learned
* How to easily build an image classifier to discern between blue jays, mockingbirds, and shrikes. 

I chose the first two bird types based on my familiarity and their Cool Local Bird ranks. I was going to select a cardinal as a third class, but thought it would be too easy to detect the class based solely on color, so I searched around and found shrikes, which looked similar to mockingbirds. The classifier performed very well, which was I semi-shock because I didn't know:

* You do not always need a lot of data to build a decent model.

Even with 150 examples of each of three classes (before train/validation split), there were only 3 misclassifications on the validation set. This shows how powerful data augmentation can be, but also helps dispel the notion that you need tons of data to do anything reasonable.

* FastAI has some powerful tools that I need to learn.

1. Easily display the examples with the highest loss and lowest confidence to see if I can interpret possible deficiencies with `interp.plot_top_losses()`.
2. Clean up the dataset with `ImageClassifierCleaner`, manually going through some of the examples to change labels or remove them, and then effect the changes with a couple simple for-loops.
3. Use `verify_images` to clean up corrupted files easily.
4. Use `DataBlock` to define the structure of the problem and implement useful transformations for the data.

* Think about how to normalize images.
  
Squashing a large image into a smaller one with simple scaling or adding cropped parts of small images is maybe not the best way to make sure the inputs are all "from the same distribution". You can use `RandomResizedCrop` to maintain original image quality and also artificially expand the size of your training set.

* Other things:
  * How to easily build a dataset with the Bing Image API.
  * The usefulness of Path (which I only recently learned about) was really demonstrated nicely.
  * The Drivetrain Approach.
  * Creating a Notebook App with widgets and Voilà. (Admittedly, I'm still trying to get the latter to work.)

## Things I Had to Troubleshoot
* How do you find the Bing Image Search key?

Solution: I signed up for the free Azure thing, went to to the [Bing Image Search API](https://azure.microsoft.com/en-us/services/cognitive-services/bing-image-search-api/), and just had to click to add Bing Search APIs v7 to my subscription. Once I did this it brought up a page with the keys and endpoints.

* I am using my local computer, with Windows, to run the notebooks. So, I have run into (standard) issues. Namely, I saw `RuntimeError: cuda runtime error (801) : operation not supported at...` when I first attempted to fine tune the learner `learn.fine_tune(4)`.

Solution: When you google the error [this issue page](https://github.com/fastai/fastbook/issues/85) points you to the forum, but also usefully mentions the "need to set num_workers=0 when creating a DataLoaders because Pytorch multiprocessing does not work on Windows." So, doing this when you define the dataloaders `dls = bears.dataloaders(path, num_workers=0)` cleared it up for me.

* Some minor formatting issues with `interp.plot_top_losses(5, nrows=1)`. The text above the images was overlapping, because my class names were a bit long.

Simple fix was to set `nrows=5`.
