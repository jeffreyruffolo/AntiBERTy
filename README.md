# AntiBERTy

Antibody-specific transformer language model pre-trained on 558M natural antibody sequences.

## Usage

```
from antiberty import AntiBERTy, get_weights

antiberty = AntiBERTy.from_pretrained(get_weights()) 
```

## Citing this work

```bibtex
@article{ruffolo2021deciphering,
    title = {Deciphering antibody affinity maturation with language models and weakly supervised learning},
    author = {Ruffolo, Jeffrey A and Gray, Jeffrey J and Sulam, Jeremias},
    journal = {arXiv preprint arXiv:2112.07782},
    year= {2021}
}
```