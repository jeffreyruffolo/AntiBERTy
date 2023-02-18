# AntiBERTy
Official repository for AntiBERTy, an antibody-specific transformer language model pre-trained on 558M natural antibody sequences, as described in [Deciphering antibody affinity maturation with language models and weakly supervised learning](https://arxiv.org/abs/2112.07782).


## Setup
To use AntiBERTy, install via pip:
```bash
pip install antiberty
```

Alternatively, you can clone this repository and install the package locally:
```bash
$ git clone git@github.com:jeffreyruffolo/AntiBERTy.git 
$ pip install AntiBERTy
```

## Usage

### Embeddings

To use AntiBERTy to generate sequence embeddings, use the `embed` function. The output is a list of embedding tensors, where each tensor is the embedding for the corresponding sequence. Each embedding has dimension `[(Length + 2) x 512]`.

```python
from antiberty import AntiBERTyRunner

antiberty = AntiBERTyRunner()

sequences = [
    "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK",
]
embeddings = antiberty.embed(sequences)
```

To access the attention matrices, pass the `return_attention` flag to the `embed` function. The output is a list of attention matrices, where each matrix is the attention matrix for the corresponding sequence. Each attention matrix has dimension `[Layer x Heads x (Length + 2) x (Length + 2)]`.

```python
from antiberty import AntiBERTyRunner

antiberty = AntiBERTyRunner()

sequences = [
    "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK",
]
embeddings, attentions = antiberty.embed(sequences, return_attention=True)
```

The `embed` function can also be used with masked sequences. Masked residues should be indicated with underscores.

### Classification
To use AntiBERTy to predict the species and chain type of sequences, use the `classify` function. The output is two lists of classifications for each sequences.

```python
from antiberty import AntiBERTyRunner

antiberty = AntiBERTyRunner()

sequences = [
    "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK",
]
species_preds, chain_preds = antiberty.classify(sequences)
```

The `classify` function can also be used with masked sequences. Masked residues should be indicated with underscores.

### Mask prediction
To use AntiBERTy to predict the identity of masked residues, use the `fill_masks` function. Masked residues should be indicated with underscores. The output is a list of filled sequences, corresponding to the input masked sequences.

```python
from antiberty import AntiBERTyRunner

antiberty = AntiBERTyRunner()

sequences = [
    "____VQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGN_NYAQKFQERVTITRDM__STAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFD____GTMVTVS",
    "DVVMTQTPFSLPV__GDQASISCRSSQSLVHSNGNTY_HWYLQKPGQSPKLLIYKVSNRFSGVPDRFSG_GSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGG__KLEIK",
]
filled_sequences = antiberty.fill_masks(sequences)
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