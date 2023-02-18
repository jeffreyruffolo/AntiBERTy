import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, ClassVar

import itertools

import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForSpeciesChain(DataCollatorForLanguageModeling):
    """
    Data collator used for training AntiBERTy model by MLM and species + chain classification.
    """
    SPECIES_TO_LABEL: ClassVar[dict] = {
        "Camel": 0,
        "human": 1,
        "HIS-mouse": 2,
        "mouse_Balb/c": 2,
        "mouse_BALB/c": 2,
        "mouse_C57BL/6": 2,
        "mouse_C57BL/6J": 2,
        "mouse_Ighe/e": 2,
        "mouse_Ighg/g": 2,
        "mouse_Igh/wt": 2,
        "mouse_outbred": 2,
        "mouse_outbred/C57BL/6": 2,
        "mouse_RAG2-GFP/129Sve": 2,
        "mouse_Swiss-Webster": 2,
        "rabbit": 3,
        "rat": 4,
        "rat_SD": 4,
        "rhesus": 5
    }
    NUM_SPECIES: ClassVar[int] = 6
    CHAIN_TO_LABEL: ClassVar[dict] = {"Heavy": 0, "Light": 1}
    NUM_CHAINS: ClassVar[int] = 2

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    missing_res_token: str = "[UNK]"

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor,
                                   Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids, batch = self.process_examples(examples)
            batched_input_ids = self.tokenizer.pad(
                input_ids,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            assert NotImplementedError

        # Update batch with padded input_ids and attention_mask
        batch.update(batched_input_ids)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask)

        return batch

    def process_examples(self, examples):
        example_species_dict = {
            i: self.SPECIES_TO_LABEL[e["Species"]]
            for i, e in enumerate(examples)
        }
        species_example_dict = {s: [] for s in range(self.NUM_SPECIES)}
        for i, s in example_species_dict.items():
            species_example_dict[s].append(i)

        examples_ = []
        for i, example in enumerate(examples):
            seq = " ".join(list(example["seq"])).replace(
                "-", self.missing_res_token)

            input_ids = self.tokenizer(seq)["input_ids"]
            species_label = self.SPECIES_TO_LABEL[example["Species"]]
            chain_label = self.CHAIN_TO_LABEL[example["Chain"]]

            examples_.append(
                dict(input_ids=input_ids,
                     species_label=species_label,
                     chain_label=chain_label))

        examples = examples_
        batch = {
            key: [example[key] for example in examples]
            for key in examples[0].keys()
        }
        batch["species_label"] = torch.tensor(batch["species_label"])
        batch["chain_label"] = torch.tensor(batch["chain_label"])

        input_ids = {"input_ids": batch["input_ids"]}

        return input_ids, batch
