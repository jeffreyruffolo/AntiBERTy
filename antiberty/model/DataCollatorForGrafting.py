import random

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, ClassVar

import itertools

import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForCDRGrafting(DataCollatorForLanguageModeling):
    """
    Data collator used for training AntiBERTy model by MLM and random grafting.
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
    graft_chance: float = 0.1

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
            input_ids = example["input_ids"]
            species_label = self.SPECIES_TO_LABEL[example["Species"]]
            chain_label = self.CHAIN_TO_LABEL[example["Chain"]]
            graft_label = 0

            examples_.append(
                dict(input_ids=input_ids,
                     species_label=species_label,
                     chain_label=chain_label,
                     graft_label=graft_label))

        examples = examples_
        batch = {
            key: [example[key] for example in examples]
            for key in examples[0].keys()
        }
        batch["species_label"] = torch.tensor(batch["species_label"])
        batch["chain_label"] = torch.tensor(batch["chain_label"])
        batch["graft_label"] = torch.tensor(batch["graft_label"])

        input_ids = {"input_ids": batch["input_ids"]}

        return input_ids, batch