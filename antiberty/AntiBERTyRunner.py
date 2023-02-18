import os

import torch
import transformers
from tqdm import tqdm

import antiberty
from antiberty import AntiBERTy
from antiberty.utils.general import exists

project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
trained_models_dir = os.path.join(project_path, 'trained_models')

CHECKPOINT_PATH = os.path.join(trained_models_dir, 'AntiBERTy_md_smooth')
VOCAB_FILE = os.path.join(trained_models_dir, 'vocab.txt')

LABEL_TO_SPECIES = {
    0: "Camel",
    1: "Human",
    2: "Mouse",
    3: "Rabbit",
    4: "Rat",
    5: "Rhesus"
}
LABEL_TO_CHAIN = {0: "Heavy", 1: "Light"}


class AntiBERTyRunner():
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = AntiBERTy.from_pretrained(CHECKPOINT_PATH).to(self.device)
        self.model.eval()

        self.tokenizer = transformers.BertTokenizer(vocab_file=VOCAB_FILE,
                                                    do_lower_case=False)

    def embed(self, sequences, hidden_layer=-1, return_attention=False):
        """
        Embed a list of sequences.

        Args:
            sequences (list): list of sequences
            hidden_layer (int): which hidden layer to use (0 to 8)
            return_attention (bool): whether to return attention matrices

        Returns:
            list(torch.Tensor): list of embeddings (one tensor per sequence)

        """
        sequences = [list(s) for s in sequences]
        for s in sequences:
            for i, c in enumerate(s):
                if c == "_":
                    s[i] = "[MASK]"

        sequences = [" ".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attention,
            )

        # gather embeddings
        embeddings = outputs.hidden_states
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = list(embeddings.detach().cpu())

        for i, a in enumerate(attention_mask):
            embeddings[i] = embeddings[i][:, a == 1]

        if exists(hidden_layer):
            for i in range(len(embeddings)):
                embeddings[i] = embeddings[i][hidden_layer]

        # gather attention matrices
        if return_attention:
            attentions = outputs.attentions
            attentions = torch.stack(attentions, dim=1)
            attentions = list(attentions.detach().cpu())

            for i, a in enumerate(attention_mask):
                attentions[i] = attentions[i][:, :, a == 1]
                attentions[i] = attentions[i][:, :, :, a == 1]

            return embeddings, attentions

        return embeddings

    def fill_masks(self, sequences):
        """
        Fill in the missing residues in a list of sequences. Each missing token is
        represented by an underscore character.

        Args:
            sequences (list): list of sequences with _ (underscore) tokens

        Returns:
            list: list of sequences with missing residues filled in
        """
        sequences = [list(s) for s in sequences]
        for s in sequences:
            for i, c in enumerate(s):
                if c == "_":
                    s[i] = "[MASK]"

        sequences = [" ".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
            )
            logits = outputs.prediction_logits
            logits[:, :, self.tokenizer.all_special_ids] = -float("inf")

        predicted_tokens = torch.argmax(logits, dim=-1)
        tokens[tokens == self.tokenizer.mask_token_id] = predicted_tokens[
            tokens == self.tokenizer.mask_token_id]

        predicted_seqs = self.tokenizer.batch_decode(
            tokens,
            skip_special_tokens=True,
        )
        predicted_seqs = [s.replace(" ", "") for s in predicted_seqs]

        return predicted_seqs

    def classify(self, sequences):
        """
        Classify a list of sequences by species and chain type. Sequences may contain
        missing residues, which are represented by an underscore character.

        Args:
            sequences (list): list of sequences

        Returns:
            list: list of species predictions
            list: list of chain type predictions
        """

        sequences = [list(s) for s in sequences]
        for s in sequences:
            for i, c in enumerate(s):
                if c == "_":
                    s[i] = "[MASK]"

        sequences = [" ".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
            )
            species_logits = outputs.species_logits
            chain_logits = outputs.chain_logits

        species_preds = torch.argmax(species_logits, dim=-1)
        chain_preds = torch.argmax(chain_logits, dim=-1)

        species_preds = [LABEL_TO_SPECIES[p.item()] for p in species_preds]
        chain_preds = [LABEL_TO_CHAIN[p.item()] for p in chain_preds]

        return species_preds, chain_preds