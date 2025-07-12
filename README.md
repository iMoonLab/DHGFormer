<div align="center">
<h2>DHGFormer: Dynamic Hierarchical Graph Transformer for Disorder Brain Disease Diagnosis</h2>

<p align="center">
  <b>Rundong Xue, Hao Hu, Zeyu Zhang, Xiangmin Han<sup>*</sup>, Juan Wang, Yue Gao, Shaoyi Du<sup>*</sup></b>
</p>

Accepted by _**MICCAI 2025**_
</div>

## Overview
<div align="center">
    <img src="assets/pipeline.png">
</div>


**Figure 1. The framework of the proposed DHGFormer.**


**_Abstract -_** The functional brain network exhibits a hierarchical characterized organization, balancing localized specialization with global integration through multi-scale hierarchical connectivity. While graph-based methods have advanced brain network analysis, conventional graph neural networks (GNNs) face interpretational limitations when modeling functional connectivity (FC) that encodes excitatory/inhibitory distinctions, often resorting to oversimplified edge weight transformations. Existing methods usually inadequately represent the brain's hierarchical organization, potentially missing critical information about multi-scale feature interactions. To address these limitations, we propose a novel brain network generation and analysis approach--Dynamic Hierarchical Graph Transformer (DHGFormer). Specifically, our method introduces an FC-inspired dynamic attention mechanism that adaptively encodes brain excitatory/inhibitory connectivity patterns into transformer-based representations, enabling dynamic adjustment of the functional brain network. Furthermore, we design hierarchical GNNs that consider prior functional subnetwork knowledge to capture intra-subnetwork homogeneity and inter-subnetwork heterogeneity, thereby enhancing GNN performance in brain disease diagnosis tasks. Extensive experiments on the ABIDE and ADNI datasets demonstrate that DHGFormer consistently outperforms state-of-the-art methods in diagnosing neurological disorders.

## Get Started
### 1. Data Preparation
Download the ABIDE dataset from [here](https://drive.google.com/file/d/14UGsikYH_SQ-d_GvY2Um2oEHw3WNxDY3/view?usp=sharing).

### 2. Usage
Run the following command to train the model.
```bash
python main.py --config_filename setting/abide_DHGFormer.yaml
```

## Cite our work
```bibtex
@inproceedings{xue2025dhgformer,
  title = {DHGFormer: Dynamic Hierarchical Graph Transformer for Disorder Brain Disease Diagnosis},
  author = {Xue, Rundong and Hu, Hao and Zhang, Zeyu and Han, Xiangmin and Wang, Juan and Gao, Yue and Du, Shaoyi},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year = {2025},
}
```

## License
The source code is free for research and education use only. Any comercial use should get formal permission first.

This repo benefits from [FBNETGEN](https://github.com/Wayfear/FBNETGEN).  Thanks for their wonderful works.
