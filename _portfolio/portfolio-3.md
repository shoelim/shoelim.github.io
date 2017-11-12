---
title: "Theoretical analyses in Machine Learning"
excerpt: "<img src='/images/circle.png'> A mathematician's take on machine learning."
collection: portfolio
---

Several recent works have empirically observed that Convolutional Neural Nets (CNNs) are (approximately) invertible. To understand this approximate invertibility phenomenon and how to leverage it more effectively, we focus on a theoretical explanation and develop a mathematical model of sparse signal recovery that is consistent with CNNs with random weights. We give an exact connection to a particular model of model-based compressive sensing (and its recovery algorithms) and random-weight CNNs. We show empirically that several learned networks are consistent with our mathematical analysis and then demonstrate that with such a simple theoretical framework, we can obtain reasonable re- construction results on real images. We also discuss gaps between our model assumptions and the CNN trained for classification in practical scenarios.

- __Paper:__ Anna C. Gilbert, Yi Zhang, Kibok Lee, Yuting Zhang, Honglak Lee, ["Towards Understanding the Invertibility of Convolutional Neural Networks"](https://arxiv.org/abs/1705.08664), IJCAI 2017.
- __Slides:__ <a href="https://annacgilbert.github.io/files/inv-cnn-ijcai2017-slides.pptx">Slides (pptx) for IJCAI 2017 presentation.</a>
- __Poster:__ <a href="https://annacgilbert.github.io/files/inv-cnn-ijcai2017-poster.pptx">Poster (pptx) for IJCAI 2017 presentation.</a>