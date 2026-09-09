from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertModel, BertPreTrainedModel, ModelOutput

from antiberty.utils.general import exists

NUM_SPECIES = 6
NUM_CHAINS = 2
NUM_GRAFTS = 2


class AntiBERTyHeads(nn.Module):
    """
    Classification heads for AntiBERTy model.
    """

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.species = nn.Linear(config.hidden_size, NUM_SPECIES)
        self.chain = nn.Linear(config.hidden_size, NUM_CHAINS)
        self.graft = nn.Linear(config.hidden_size, NUM_GRAFTS)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        species_score = self.species(pooled_output)
        chain_score = self.chain(pooled_output)
        graft_score = self.graft(pooled_output)
        return prediction_scores, species_score, chain_score, graft_score


@dataclass
class AntiBERTyOutput(ModelOutput):
    """
    Output type of for AntiBERTy model.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    species_logits: torch.FloatTensor = None
    chain_logits: torch.FloatTensor = None
    graft_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class AntiBERTy(BertPreTrainedModel):
    """
    BERT model for antibody sequences, with classification heads
    for species, chain type, and presence of grafting
    """

    # The MLM decoder shares its weight with the input embeddings and its bias with
    # `cls.predictions.bias`; the checkpoint stores one copy of each.
    _tied_weights_keys = {
        "cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
        "cls.predictions.decoder.bias": "cls.predictions.bias",
    }

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = AntiBERTyHeads(config)

        self.post_init()

        self.num_species = NUM_SPECIES
        self.num_chains = NUM_CHAINS
        self.num_grafts = NUM_GRAFTS

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        species_label=None,
        chain_label=None,
        graft_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not exists(return_dict):
            return_dict = getattr(self.config, "return_dict", True)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, species_score, chain_score, graft_score = self.cls(sequence_output, pooled_output)

        b = input_ids.shape[0]

        total_loss, masked_lm_loss, species_loss, chain_loss, graft_loss = None, None, None, None, None
        if exists(labels):
            mlm_loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = mlm_loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        def class_loss(logits, labels, num_classes):
            # inverse-frequency class weights over the batch
            freqs = torch.bincount(labels, minlength=num_classes).clamp(min=1)
            weights = b / (freqs * num_classes)
            return nn.CrossEntropyLoss(weight=weights.to(logits.dtype))(logits.view(-1, num_classes), labels.view(-1))

        if exists(species_label):
            species_loss = class_loss(species_score, species_label, self.num_species)

        if exists(chain_label):
            chain_loss = class_loss(chain_score, chain_label, self.num_chains)

        if exists(graft_label):
            graft_loss = class_loss(graft_score, graft_label, self.num_grafts)

        losses = [l for l in (masked_lm_loss, species_loss, chain_loss, graft_loss) if exists(l)]
        total_loss = sum(losses) if len(losses) > 0 else None

        if not return_dict:
            output = (prediction_scores, species_score, chain_score, graft_score) + outputs[2:]
            return ((total_loss,) + output) if exists(total_loss) else output

        return AntiBERTyOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            species_logits=species_score,
            chain_logits=chain_score,
            graft_logits=graft_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
