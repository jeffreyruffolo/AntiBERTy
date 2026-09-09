"""
Tokenizer for AntiBERTy: a 25-token vocabulary (5 special tokens + 20 amino acids), one token per
residue, with ``[CLS]``/``[SEP]`` added around each sequence and right-padding to the longest
sequence in a batch.
"""

from typing import Dict, List, Sequence, Union

import torch

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
MASK_CHAR = "_"
VOCAB: List[str] = [PAD, UNK, CLS, SEP, MASK, *"ACDEFGHIKLMNPQRSTVWY"]
MAX_LEN = 512  # positional-embedding limit of the model, including [CLS]/[SEP]


class AntiBERTyTokenizer:
    def __init__(self, vocab_file: str = None):
        """
        :param vocab_file: optional vocabulary file (one token per line); defaults to the built-in
            AntiBERTy vocabulary.
        """
        if vocab_file is None:
            self.vocab = list(VOCAB)
        else:
            with open(vocab_file) as f:
                self.vocab = [line.strip() for line in f if len(line.strip()) > 0]
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.vocab)}

        self.pad_token_id = self.token_to_id[PAD]
        self.unk_token_id = self.token_to_id[UNK]
        self.cls_token_id = self.token_to_id[CLS]
        self.sep_token_id = self.token_to_id[SEP]
        self.mask_token_id = self.token_to_id[MASK]
        self.all_special_ids = [
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
        ]
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.vocab)

    def tokenize(self, sequence: Union[str, Sequence[str]]) -> List[str]:
        """
        Split a sequence into residue tokens. A string is read one residue per character, with
        ``_`` for a masked position (whitespace-separated strings are split into tokens instead);
        an iterable of tokens is used as is. Residues are upper-cased; anything not in the
        vocabulary becomes ``[UNK]``.
        """
        if isinstance(sequence, str):
            sequence = sequence.strip()
            if any(c.isspace() for c in sequence):
                sequence = sequence.split()
            elif "[" in sequence:
                raise ValueError(f"Use '{MASK_CHAR}' for masked residues in plain strings (got {sequence!r}).")
            else:
                sequence = list(sequence)

        tokens = []
        for t in sequence:
            if t == MASK_CHAR or t == MASK:
                tokens.append(MASK)
            else:
                t = t.upper()
                tokens.append(t if t in self.token_to_id else UNK)

        if len(tokens) == 0:
            raise ValueError("Cannot tokenize an empty sequence.")

        return tokens

    def encode(self, sequence, add_special_tokens: bool = True) -> List[int]:
        ids = [self.token_to_id[t] for t in self.tokenize(sequence)]
        n_residues = len(ids)
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        if len(ids) > self.max_len:
            raise ValueError(
                f"Sequence of length {n_residues} exceeds the AntiBERTy limit of {self.max_len - 2} residues. "
                "AntiBERTy expects antibody variable domains (typically 100-130 residues)."
            )

        return ids

    def __call__(self, sequences: Union[str, Sequence]) -> Dict[str, torch.Tensor]:
        """
        Batch-encode sequences into ``input_ids`` and ``attention_mask`` tensors, right-padded
        with ``[PAD]`` to the longest sequence in the batch.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        encoded = [self.encode(s) for s in sequences]
        max_len = max(len(e) for e in encoded)

        input_ids, attention_mask = [], []
        for e in encoded:
            pad = max_len - len(e)
            input_ids.append(e + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(e) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [self.vocab[int(i)] for i in ids]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [t for t, i in zip(tokens, ids) if int(i) not in self.all_special_ids]

        return "".join(tokens)

    def batch_decode(self, batch_ids, skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]
