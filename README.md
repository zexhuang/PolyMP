# Learning Geometric Invariant Features for Classification of Vector Polygons with Graph Message-passing Neural Network

## Overview

This is the official code repository of paper *Learning Geometric Invariant Features for Classification of Vector Polygons with Graph Message-passing Neural Network*.

## Abstract

Geometric shape classification of vector polygons remains a challenging task in spatial analysis. Previous studies have primarily focused on deep learning approaches for rasterized vector polygons, while the study of discrete polygon representations and corresponding learning methods remains underexplored. In this study, we investigate a graph-based representation of vector polygons and propose a simple graph message-passing framework, PolyMP, along with its densely self-connected variant, PolyMP-DSC, to learn more expressive and robust latent representations of polygons. This framework hierarchically captures self-looped graph information and learns geometric-invariant features for polygon shape classification. Through extensive experiments, we demonstrate that combining a permutation-invariant graph message-passing neural network with a densely self-connected mechanism achieves robust performance on benchmark datasets, including synthetic glyphs and real-world building footprints, outperforming several baseline methods. Our findings indicate that PolyMP and PolyMP-DSC effectively capture expressive geometric features that remain invariant under common transformations, such as translation, rotation, scaling, and shearing, while also being robust to trivial vertex removals. Furthermore, we highlight the strong generalization ability of the proposed approach, enabling the transfer of learned geometric features from synthetic glyph polygons to real-world building footprints.  

## Requirements

* python=3.8
* pytorch=1.10
* shapely
* numpy
* pandas
* geopandas
* freetype
* sklearn
