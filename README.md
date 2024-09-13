# Adaptive Semantic Consistency for Cross-domain Few-shot Classification
This is the official code for https://arxiv.org/pdf/2308.00727.

If you find this repo useful for your research, please consider citing this paper  
```
@article{lu2023adaptive,
  title={Adaptive Semantic Consistency for Cross-domain Few-shot Classification},
  author={Lu, Hengchu and Shao, Yuanjie and Wang, Xiang and Gao, Changxin},
  journal={arXiv preprint arXiv:2308.00727},
  year={2023}
}
```
# Dataset & Pretrained Mdoel Download
Please follow [here](https://github.com/hytseng0509/CrossDomainFewShot#datasets) to download dataset and pretrained model

# Running

* To run contrastive finetuning on `EuroSAT` data (default target domain), run ```python Supcon_ASC.py```, with 5-way 5-shot as default setting

# Acknowlegements

Our code is based on [On the Importance of Distractors for Few-Shot Classification](https://github.com/quantacode/Contrastive-Finetuning).

