---
title: 'Choosing Astro'
description: 'Why I chose Astro for this site'
pubDate: 'Mar 15 2026'
heroImage: '../../assets/blog-placeholder-3.jpg'
---

The previous iteration of this site used Quarto. Highly recommend that still because you can write posts in markdown and notebooks and end up with a good-looking site.

I wanted more flexibility to try experiments and use this for personal utilities, so I decided to go to JS-land.

Astro seems like a good choice because of its [islands architecture](https://docs.astro.build/en/concepts/islands/). Basically: render the easy things immediately, let parts of your site work independently, and embrace multi-page apps. Another nice feature is that it let's you mix in multiple JS frameworks if you like to experiment.

Finally, it easily integrates markdown, which is the bulk of how I share stuff.