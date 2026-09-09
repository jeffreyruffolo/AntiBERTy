from typing import List, Optional, Sequence, Tuple, Union

import torch
import transformers
from tqdm import tqdm

from antiberty.model.AntiBERTy import AntiBERTy
from antiberty.tokenizer import MASK_CHAR, AntiBERTyTokenizer
from antiberty.utils.general import exists
from antiberty.utils.get_weights import get_weights

LABEL_TO_SPECIES = {0: "Camel", 1: "Human", 2: "Mouse", 3: "Rabbit", 4: "Rat", 5: "Rhesus"}
LABEL_TO_CHAIN = {0: "Heavy", 1: "Light"}
LABEL_TO_GRAFT = {0: "Natural", 1: "Grafted"}

SPECIES_TO_LABEL = {v: k for k, v in LABEL_TO_SPECIES.items()}
CHAIN_TO_LABEL = {v: k for k, v in LABEL_TO_CHAIN.items()}

Sequences = Union[str, Sequence[str]]


def resolve_device(device=None) -> torch.device:
    """
    Resolve a user-supplied device (str, torch.device or None) to a torch.device.
    Defaults to the first CUDA device if available, otherwise CPU.
    """
    if exists(device):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_list(sequences: Sequences) -> List[str]:
    return [sequences] if isinstance(sequences, str) else list(sequences)


class AntiBERTyRunner:
    def __init__(self, device=None, checkpoint_path: Optional[str] = None):
        """
        Load the pre-trained AntiBERTy model.

        :param device: torch device (e.g. "cpu", "cuda:1", "mps"). Defaults to CUDA if available.
        :param checkpoint_path: directory containing config.json and model.safetensors. By default the
            weights are taken from $ANTIBERTY_WEIGHTS_DIR, the package, or downloaded from the
            Hugging Face Hub (https://huggingface.co/jeffruffolo/AntiBERTy) and cached.
        """
        self.device = resolve_device(device)
        self.checkpoint_path = get_weights(checkpoint_path)

        progress_bars = transformers.utils.logging.is_progress_bar_enabled()
        transformers.utils.logging.disable_progress_bar()
        try:
            # eager attention is required for `output_attentions`; SDPA does not return attention matrices
            self.model = AntiBERTy.from_pretrained(self.checkpoint_path, attn_implementation="eager")
        finally:
            if progress_bars:
                transformers.utils.logging.enable_progress_bar()
        self.model.to(self.device).eval()

        self.tokenizer = AntiBERTyTokenizer()

    def _forward(
        self, sequences: Sequences, **kwargs
    ) -> Tuple[transformers.utils.ModelOutput, torch.Tensor, torch.Tensor]:
        """Tokenize, move to the model device and run a no-grad forward pass."""
        tokenizer_out = self.tokenizer(sequences)
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=tokens, attention_mask=attention_mask, **kwargs)

        return outputs, tokens, attention_mask

    def embed(self, sequences: Sequences, hidden_layer: Optional[int] = -1, return_attention: bool = False):
        """
        Embed sequences.

        :param sequences: sequences; masked residues are written as ``_``.
        :param hidden_layer: which of the 9 hidden states to return (0 = embedding layer output,
            -1 = last encoder layer), or None for all layers.
        :param return_attention: also return the attention matrices.
        :return: list of ``(L+2) x 512`` tensors (``9 x (L+2) x 512`` with ``hidden_layer=None``),
            one per sequence, including the ``[CLS]``/``[SEP]`` positions. With ``return_attention``,
            a tuple ``(embeddings, attentions)`` where each attention tensor is
            ``layers x heads x (L+2) x (L+2)``.
        """
        outputs, _, attention_mask = self._forward(
            sequences, output_hidden_states=True, output_attentions=return_attention
        )

        hidden = torch.stack(outputs.hidden_states, dim=1)  # b x layers x L x d
        embeddings = []
        for i, keep in enumerate(attention_mask.bool()):
            e = hidden[i][:, keep]
            embeddings.append(e if hidden_layer is None else e[hidden_layer])

        if not return_attention:
            return embeddings

        attn = torch.stack(outputs.attentions, dim=1)  # b x layers x heads x L x L
        attentions = [attn[i][:, :, keep][:, :, :, keep] for i, keep in enumerate(attention_mask.bool())]

        return embeddings, attentions

    def fill_masks(self, sequences: Sequences) -> List[str]:
        """
        Fill masked residues (``_``) with the most likely amino acid.

        :return: list of completed sequences; unmasked positions are returned unchanged.
        """
        sequences = _as_list(sequences)
        outputs, tokens, _ = self._forward(sequences)

        logits = outputs.prediction_logits
        logits[:, :, self.tokenizer.all_special_ids] = -float("inf")
        predicted = logits.argmax(dim=-1)

        filled = []
        for seq, tok, pred in zip(sequences, tokens, predicted):
            residues = self.tokenizer.tokenize(seq)
            is_mask = (tok[1 : len(residues) + 1] == self.tokenizer.mask_token_id).tolist()
            pred_tokens = self.tokenizer.convert_ids_to_tokens(pred[1 : len(residues) + 1])
            chars = list(seq.strip()) if len(seq.strip()) == len(residues) else residues
            filled.append("".join(p if m else c for c, m, p in zip(chars, is_mask, pred_tokens)))

        return filled

    def classify(
        self,
        sequences: Sequences,
        species_label: Optional[str] = None,
        chain_label: Optional[str] = None,
        graft_label=None,
        return_graft: bool = False,
    ):
        """
        Predict species and chain type, and optionally whether the sequence looks like a CDR graft
        (e.g. a humanized antibody). Masked residues (``_``) are allowed.

        :return: ``(species, chains)`` lists with one entry per sequence, or
            ``(species, chains, grafts)`` with ``return_graft=True`` where each graft entry is
            "Natural" or "Grafted". If any of ``species_label`` (e.g. "Human"), ``chain_label``
            ("Heavy"/"Light") or ``graft_label`` (0/1) is given, a single sequence is scored against
            those labels instead and a dict of log-likelihoods (``species_ll``, ``chain_ll``,
            ``graft_ll``) is returned.
        """
        sequences = _as_list(sequences)
        outputs, _, _ = self._forward(sequences)
        species_logits, chain_logits, graft_logits = outputs.species_logits, outputs.chain_logits, outputs.graft_logits

        if exists(species_label) or exists(chain_label) or exists(graft_label):
            if len(sequences) != 1:
                raise ValueError("Label scoring supports exactly one sequence at a time.")
            out_dict = {}
            for name, label, mapping, logits in (
                ("species_ll", species_label, SPECIES_TO_LABEL, species_logits),
                ("chain_ll", chain_label, CHAIN_TO_LABEL, chain_logits),
                ("graft_ll", graft_label, None, graft_logits),
            ):
                if not exists(label):
                    continue
                idx = int(label) if mapping is None else mapping[label]
                target = torch.tensor([idx], device=self.device)
                out_dict[name] = -torch.nn.functional.cross_entropy(logits, target).item()

            return out_dict

        species = [LABEL_TO_SPECIES[p] for p in species_logits.argmax(dim=-1).tolist()]
        chains = [LABEL_TO_CHAIN[p] for p in chain_logits.argmax(dim=-1).tolist()]
        if not return_graft:
            return species, chains

        grafts = [LABEL_TO_GRAFT[p] for p in graft_logits.argmax(dim=-1).tolist()]

        return species, chains, grafts

    def graft_probability(self, sequences: Sequences) -> torch.Tensor:
        """Probability that each sequence contains grafted CDRs (e.g. is humanized); shape ``(N,)``."""
        outputs, _, _ = self._forward(_as_list(sequences))

        return torch.softmax(outputs.graft_logits, dim=-1)[:, 1]

    def pseudo_log_likelihood(
        self,
        sequences: Sequences,
        batch_size: Optional[int] = 64,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Pseudo log-likelihood of each sequence: every position is masked in turn and the
        log-probability of the true residue is averaged over positions.

        :param batch_size: masked copies scored per forward pass (None for all at once; memory grows
            with L^3).
        :param verbose: show a progress bar over sequences.
        :return: tensor of shape ``(N,)``.
        """
        sequences = _as_list(sequences)
        plls = []
        for s in tqdm(sequences, disable=not verbose, leave=False):
            residues = self.tokenizer.tokenize(s)
            labels = torch.tensor(self.tokenizer.encode(residues)[1:-1], device=self.device)
            valid = torch.tensor([i not in self.tokenizer.all_special_ids for i in labels.tolist()], device=self.device)
            if not valid.any():
                raise ValueError(f"Sequence {s!r} has no standard residues to score.")

            masked = [residues[:i] + [MASK_CHAR] + residues[i + 1 :] for i in range(len(residues))]
            step = len(masked) if batch_size is None else batch_size
            logits = []
            for start in range(0, len(masked), step):
                outputs, _, _ = self._forward(masked[start : start + step])
                logits.append(outputs.prediction_logits)
            logits = torch.cat(logits, dim=0)[:, 1:-1]  # copies x L x vocab, without [CLS]/[SEP]
            logits[:, :, self.tokenizer.all_special_ids] = -float("inf")

            masked_logits = logits[
                torch.arange(len(residues), device=self.device), torch.arange(len(residues), device=self.device)
            ]
            log_probs = torch.log_softmax(masked_logits[valid], dim=-1)
            plls.append(log_probs.gather(1, labels[valid, None]).mean())

        return torch.stack(plls, dim=0)
