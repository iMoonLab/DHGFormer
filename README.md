# DHGFormer
<h2 align="center"> [MICCAI 2025] DHGFormer: Dynamic Hierarchical Graph Transformer for Disorder Brain Disease Diagnosis</h2>

<p align="center">
  <b>Rundong Xue, Hao Hu, Zeyu Zhang, Xiangmin Han<sup>*</sup>, Juan Wang, Yue Gao, Shaoyi Du<sup>*</sup></b>
</p>


## Overview
<div>
    <img src="figures/pipeline.png" width="96%" height="96%">
</div>

**Figure 1. The framework of the proposed HGST.**


**_Abstract -_** The functional brain network exhibits a hierarchical characterized organization, balancing localized specialization with global integration through multi-scale hierarchical connectivity. While graph-based methods have advanced brain network analysis, conventional graph neural networks (GNNs) face interpretational limitations when modeling functional connectivity (FC) that encodes excitatory/inhibitory distinctions, often resorting to oversimplified edge weight transformations. Existing methods usually inadequately represent the brain's hierarchical organization, potentially missing critical information about multi-scale feature interactions. To address these limitations, we propose a novel brain network generation and analysis approach--Dynamic Hierarchical Graph Transformer (DHGFormer). Specifically, our method introduces an FC-inspired dynamic attention mechanism that adaptively encodes brain excitatory/inhibitory connectivity patterns into transformer-based representations, enabling dynamic adjustment of the functional brain network. Furthermore, we design hierarchical GNNs that consider prior functional subnetwork knowledge to capture intra-subnetwork homogeneity and inter-subnetwork heterogeneity, thereby enhancing GNN performance in brain disease diagnosis tasks. Extensive experiments on the ABIDE and ADNI datasets demonstrate that DHGFormer consistently outperforms state-of-the-art methods in diagnosing neurological disorders.

## Cite our work
```bibtex
Coming soon...
```

## License
The source code is free for research and education use only. Any comercial use should get formal permission first.
This repo benefits from [FBNETGEN]  (https://github.com/Wayfear/FBNETGEN).  Thanks for their wonderful works.
