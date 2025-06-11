# Learning Geometric Invariant Features for Classification of Vector Polygons with Graph Message-passing Neural Network

## Overview

This is the official code repository of paper *Learning Geometric Invariant Features for Classification of Vector Polygons with Graph Message-passing Neural Network*.

## Abstract

Geometric shape classification of vector polygons remains a non-trivial learning task in spatial analysis. Previous studies mainly focus on devising deep learning approaches for representation learning of rasterized vector polygons, whereas the study of discrete representations of polygons and subsequent deep learning approaches have not been fully investigated. In this study, we investigate a graph representation of vector polygons and propose a novel graph message-passing neural network (PolyMP) to learn the geometric-invariant features for shape classification of polygons. Through extensive experiments, we show that the graph representation of polygons combined with a permutation-invariant graph message-passing neural network achieves highly robust performances on benchmark datasets (i.e., synthetic glyph and real-world building footprint datasets) as compared to baseline methods. We demonstrate that the proposed graph-based PolyMP network enables the learning of expressive geometric features invariant to geometric transformations of polygons (i.e., translation, rotation, scaling and shearing) and is robust to trivial vertex removals of polygons. We further show the strong generalizability of PolyMP, which enables generalizing the learned geometric features from the synthetic glyph polygons to the real-world building footprints.  

## Requirements

* python=3.8
* pytorch=1.10
* shapely
* numpy
* pandas
* geopandas
* freetype
* sklearn
