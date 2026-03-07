---
aliases:
- /markdown/2020/07/12/wind_nightmare
categories:
- notebooks
- virtualenv
date: '2020-07-12'
description: Headaches with virtualenvs
layout: post
title: Solving a Nightmare with a Headache
toc: true

---

# Solving a Nightmare with a Headache

**Issue**: The widgets in Jupyter Notebook were not all displaying. 

**Non-solution**: Try re-do some installs in the virtualenv. While I am at it, let's also try out pyenv.

**New Issue**: Windows does not play well with pyenv. You can get pyenv to work, but then virtualenv is no longer set up on your global environment.

**Attempted Solution**: Try to clone the pyenv-virtualenv repo into a new plugins directory of the pyenv root folder. This did not work, despite sinking about an hour into this whole mess.

**New Issue**: It's time to put Linux on my machine to try and escape some of these Windows nightmares. This is going to be a headache, but it seems like the pay off will be worth-while, given that the Fastbook also doesn't play well with Windows. I have been frustrated with doing Python things on Windows before so I had Linux available already. I will note that it took be a while to remember I already had a Linux partition and in a cascading wave of failure, I forgot my linux password, so was not able to run a needed sudo command. But, I got lucky again (and this is a yikes security-wise) and it turns out you can easily reset the password for Ubuntu. 

From here I followed [Real Python's primer on pyenv](https://realpython.com/intro-to-pyenv/#virtual-environments-and-pyenv). Setting this up was not bad and putting some lines in .bashrc virtualenv becomes much nicer to use. It allows you connect a particular environment to a folder. I created a virtualenv named `fab` (FastAI Book, short environment names are good when you're used to activating an environment every time you want to work on something) by again [following Real Python](https://realpython.com/python-virtual-environments-a-primer/) and now it is automatically activated when I am in the fastbook directory. A quick `pip install -r requirements.txt -v` got the environment running smoothly with the notebook for Chapter 1, and things that did not work 100% in Windows worked in my Linux boot nicely (at least after a `sudo apt install graphviz`):
* I could run the cells without having to change `num_workers` to 0.
* My GPU was visible to Torch (`torch.cuda.get_device_name(0)` returned the name of my GPU).
* The text example, which did not work at all on Windows, ran.
* And, the whole reason I started this endeavor: when I created the `FileUploader` widget, it appeared in the notebook.
