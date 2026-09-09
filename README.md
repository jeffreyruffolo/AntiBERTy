# AntiBERTy

Official repository for AntiBERTy, an antibody-specific transformer language model pre-trained on 558M natural antibody sequences, as described in [Deciphering antibody affinity maturation with language models and weakly supervised learning](https://arxiv.org/abs/2112.07782).

AntiBERTy is an 8-layer BERT model (512 hidden units, 8 heads, 26M parameters) trained on antibody variable-domain sequences with masked language modelling plus classification heads for species, chain type, and CDR grafting (whether the CDRs come from a different sequence than the framework, as in a humanized antibody). It provides the sequence features used by [IgFold](https://github.com/Graylab/IgFold) for structure prediction.

## Updates
```
 - Version 1.0.0
   - Support current PyTorch (>=2.0) and transformers (>=4.36, including 5.x)
   - Weights are downloaded from the Hugging Face Hub on first use (safetensors) and cached
   - Built-in tokenizer
   - `antiberty` command-line interface; explicit device selection (`device="cuda:1"`, "mps" supported)
   - Tests and CI across Python and transformers versions
```

## Installation

```bash
pip install antiberty
```

Requires Python >= 3.10, PyTorch >= 2.0 and transformers >= 4.36.

The pre-trained weights (about 100 MB) are downloaded from
[huggingface.co/jeffruffolo/AntiBERTy](https://huggingface.co/jeffruffolo/AntiBERTy) on first use
and cached in the Hugging Face cache directory (`~/.cache/huggingface`, or `$HF_HOME`). To use a
local copy instead, for example on a machine without network access, download `config.json`,
`model.safetensors` and `vocab.txt` from that repository once and point `ANTIBERTY_WEIGHTS_DIR` at
the directory, or pass `checkpoint_path` to `AntiBERTyRunner`.

To work from a clone of this repository:

```bash
git clone git@github.com:jeffreyruffolo/AntiBERTy.git
cd AntiBERTy
pip install -e .[dev]
```

## Command line

Every subcommand takes sequences as positional arguments and/or from `--fasta FILE` (record ids are
kept in the output, made unique if repeated), and accepts `--device` (`cpu`, `cuda:1`, `mps`, ...;
default CUDA if available). Masked residues are written as `_`. Sequences are upper-cased; anything
outside the 20 standard amino acids becomes `[UNK]`. Run `antiberty <command> --help` for details.

```bash
# species and chain type
antiberty classify EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS

# mean-pooled 512-d embeddings for every record in a FASTA file
antiberty embed --fasta antibodies.fasta --pool mean -o embeddings.pt

# fill masked residues
antiberty fill "QVQLQESGGGLVQAGGSLTLSCAVSG__FSNYAMG"

# pseudo log-likelihood
antiberty pll --fasta antibodies.fasta
```

### `antiberty embed`

Writes embeddings to a file, keyed by sequence id (`seq1`, `seq2`, ... for positional sequences).

| Option | Description |
| --- | --- |
| `-o`, `--output FILE` | Required. `.pt` saves a dict of tensors with `torch.save`; `.npz` saves NumPy arrays. |
| `--layer N` | Hidden layer to use, `0` (input embeddings) to `8` (last), or `all` for a `9 x ...` stack. Default `-1` (last). |
| `--pool none` | Default. `(L+2) x 512` per sequence, including the `[CLS]` and `[SEP]` positions. |
| `--pool residue` | `L x 512`, special tokens removed. |
| `--pool mean` | `512`, mean over residues (special tokens excluded). |
| `--attention` | Also save `<id>_attention`: `layers x heads x (L+2) x (L+2)` attention matrices. |

### `antiberty classify`

Prints a TSV of `id`, predicted species (Camel, Human, Mouse, Rabbit, Rat, Rhesus) and chain type
(Heavy, Light). With `--graft`, two more columns report whether the sequence looks CDR-grafted
(`Natural`/`Grafted`) and the graft probability. Masked residues are allowed.

### `antiberty fill`

Replaces each `_` with the most likely amino acid and prints the completed sequences (one per line,
or FASTA with `--fasta-out`).

### `antiberty pll`

Prints a TSV of `id` and pseudo log-likelihood: the mean over positions of the log-probability of
the true residue when that position is masked (positions holding `_` or `[UNK]` are skipped).
Higher (closer to 0) means more antibody-like. `--batch-size N` (default 64) sets how many masked
copies are scored per forward pass; lower it if memory is tight.

## Python API

```python
from antiberty import AntiBERTyRunner

antiberty = AntiBERTyRunner(
    device=None,             # "cpu", "cuda", "cuda:1", "mps", ...; default: CUDA if available, else CPU
    checkpoint_path=None,    # directory with config.json + model.safetensors; default: $ANTIBERTY_WEIGHTS_DIR or the Hub download
)
antiberty.model      # antiberty.AntiBERTy (a transformers BertPreTrainedModel) in eval mode, on antiberty.device
antiberty.tokenizer  # antiberty.tokenizer.AntiBERTyTokenizer
```

All methods take a list of sequences (or a single string). Sequences are upper-cased; `_` marks a
masked residue and any other character outside the 20 standard amino acids becomes `[UNK]`.
Sequences longer than 510 residues raise `ValueError` (the model's positional limit); AntiBERTy is
trained on variable domains of roughly 100-130 residues. Inputs are processed as one padded batch,
so very large lists should be chunked by the caller.

### Embeddings

```python
sequences = [
    "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK",
]

embeddings = antiberty.embed(sequences)                       # list of (L+2) x 512 tensors
embeddings = antiberty.embed(sequences, hidden_layer=None)    # list of 9 x (L+2) x 512 (all layers)
embeddings, attentions = antiberty.embed(sequences, return_attention=True)
# attentions: list of layers(8) x heads(8) x (L+2) x (L+2)
```

Position 0 is the `[CLS]` token and the last position `[SEP]`; slice `[1:-1]` for per-residue
features. `hidden_layer` indexes the 9 hidden states (0 is the embedding layer output, -1 the last
encoder layer). Tensors stay on `antiberty.device`.

### Classification

```python
species, chains = antiberty.classify(sequences)
# species: list of "Camel" | "Human" | "Mouse" | "Rabbit" | "Rat" | "Rhesus"
# chains:  list of "Heavy" | "Light"

species, chains, grafts = antiberty.classify(sequences, return_graft=True)
# grafts:  list of "Natural" | "Grafted" (CDRs from a different sequence than the framework, e.g. humanized)
p_graft = antiberty.graft_probability(sequences)   # tensor of shape (N,)
```

Passing `species_label`, `chain_label` and/or `graft_label` scores a single sequence against the
given labels instead, returning a dict of log-likelihoods (`species_ll`, `chain_ll`, `graft_ll`);
more than one sequence raises `ValueError`.

### Mask filling

```python
filled = antiberty.fill_masks(["QVQLQESGGGLVQAGGSLTLSCAVSG__FSNYAMG"])
# ['QVQLQESGGGLVQAGGSLTLSCAVSGFTFSNYAMG']
```

Each `_` is replaced by the highest-probability amino acid; all other positions are returned unchanged.

### Pseudo log-likelihood

```python
pll = antiberty.pseudo_log_likelihood(sequences, batch_size=64)   # tensor of shape (N,)
```

For each sequence, every position is masked in turn and the log-probability of the true residue is
averaged over the standard-residue positions. `batch_size` sets how many masked copies go through
the model at once (memory grows with the cube of the sequence length; `None` scores all positions
in one pass). `verbose=True` shows a progress bar over sequences.

### Tokenizer

`antiberty.tokenizer.AntiBERTyTokenizer` tokenizes one residue per token over the built-in 25-token
vocabulary (`[PAD] [UNK] [CLS] [SEP] [MASK]` + 20 amino acids). Whitespace-separated strings and
lists are read as token sequences, so `[MASK]` can be given explicitly in those forms.

```python
tok = antiberty.tokenizer
tok(["EVQL", "EV"])                 # {"input_ids": (2, 6) long tensor, "attention_mask": (2, 6)}
tok.encode("EVQL")                  # [2, 8, 22, 18, 14, 3]
tok.decode([2, 8, 22, 18, 14, 3])   # "EVQL"
tok.mask_token_id, tok.all_special_ids
```

### Using the model directly

`antiberty.AntiBERTy` is a `transformers` model and can be loaded and fine-tuned like any other:

```python
from antiberty import AntiBERTy
from antiberty.utils.get_weights import get_weights

model = AntiBERTy.from_pretrained(get_weights(), attn_implementation="eager")
out = model(input_ids=..., attention_mask=..., output_hidden_states=True, output_attentions=True)
out.prediction_logits, out.species_logits, out.chain_logits, out.hidden_states, out.attentions
```

Pass `labels`, `species_label`, `chain_label` and/or `graft_label` to obtain `out.loss` for
training. `attn_implementation="eager"` is required when attention matrices are needed.

## Development

```bash
pip install -e .[dev]
pytest                              # downloads the weights on first run
pre-commit install                  # ruff lint + format, whitespace and file checks
pre-commit run --all-files
```

CI runs the tests on Python 3.10 to 3.12 against transformers 4.36, 4.5x and 5.x.

## Citing this work

```bibtex
@article{ruffolo2021deciphering,
    title = {Deciphering antibody affinity maturation with language models and weakly supervised learning},
    author = {Ruffolo, Jeffrey A and Gray, Jeffrey J and Sulam, Jeremias},
    journal = {arXiv preprint arXiv:2112.07782},
    year= {2021}
}
```
