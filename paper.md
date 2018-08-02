---
title: 'UMAP: Uniform Manifold Approximation and Projection'
tags:
  - manifold learning
  - dimension reduction
  - unsupervised learning
authors:
 - name: Leland McInnes
   orcid: 0000-0003-2143-6834
   affiliation: 1
 - name: John Healy
   affiliation: 1
 - name: Nathaniel Saul
   affiliation: 2
 - name: Lukas Großberger
   affiliation: "3, 4"
affiliations:
 - name: Tutte Institute for Mathematics and Computing
   index: 1
 - name: Department of Mathematics and Statistics, Washington State University
   index: 2
 - name: Ernst Strüngmann Institute for Neuroscience in cooperation with Max Planck Society
   index: 3
 - name: Donders Institute for Brain, Cognition and Behaviour, Radboud Universiteit
   index: 4
date: 26 July 2018
bibliography: paper.bib
---

# Summary

Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique
that can be used for  visualisation similarly to t-SNE, but also for general non-linear
dimension reduction. UMAP has a rigorous mathematical foundation, but is simple to use,
with a scikit-learn compatible API. UMAP is among the fastest manifold learning
implementations available -- significantly faster than most t-SNE implementations.

UMAP supports a number of useful features, including the ability to use labels
(or partial labels) for supervised (or semi-supervised) dimension reduction,
and the ability to transform new unseen data into a pretrained embedding space.

For details of the mathematical underpinnings see [@umap_arxiv]. The implementation
can be found at [@umap_repo].

-![Fashion MNIST embedded via UMAP](images/umap_example_fashion_mnist1.png)

# References
