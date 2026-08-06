# PULSAR
Official code repository for the ICDM 2025 paper: ["PULSAR: Advancing Interval-Based Time Series Classification to State-of-the-Art Performance"](https://www.computer.org/csdl/proceedings-article/icdm/2025/959900b085/2eowio1bf2g)

## Abstract
State-of-the-art Time Series Classification (TSC) models such as HC2 and MultiRocket-Hydra reach high accuracy but rely on representations that are inherently difficult to interpret. Interval-based classifiers summarize local segments and thus provide more intuitive features. However, they are usually behind the most accurate methods. Our approach PULSAR—_**P**ooled m**U**lti-sca**L**e **S**ummaries from r**A**ndomized inte**R**vals_ significantly enhances the accuracy of interval-based approaches and is close to MultiRocket-Hydra. PULSAR first converts every series into several representations (raw, derivative, periodogram, and others). It then uses sub-series of varied length and dilation and computes simple local statistics. To build higher-order features, it partitions each sequence of local statistics at multiple depths, applies a randomly chosen set of pooling operators to every partition, and keeps all aggregates from the coarsest level. For finer partitions we apply a supervised feature selection strategy to retain only the most discriminative features. Finally, we concatenate the selected aggregates with the global ones and train an ensemble classifier. We test PULSAR on 142 UCR datasets. It outperforms all interval-based approaches by a statistically significant margin and matches the predictive performance of HC2 and MultiRocket-Hydra. PULSAR sets a new benchmark for interval-based TSC while preserving a feature structure that we can still interpret.

## Overview

The main experimental pipeline is in the notebook `main.ipynb`. It loads datasets, applies the PULSAR method, and saves the results of multiple resamples per dataset in a CSV file.

## Pre-computed Results

The full classification accuracy results for PULSAR on all 142 UCR datasets over 30 resamples, as reported in our paper, can be found in the following file:

`/PULSAR_results_142Datasets_30resamples/PULSAR_classification_accuracy.csv`

## Requirements

- Python ≥ 3.8
- Required packages (install with pip):

```bash
pip install -r requirements.txt
