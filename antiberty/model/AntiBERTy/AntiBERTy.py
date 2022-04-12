from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, ModelOutput

from antiberty.utils.general import exists


class AntiBERTyHeads(nn.Module):
    """
    Classification heads for AntiBERTy model.
    """
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.species = nn.Linear(config.hidden_size, 6)
        self.chain = nn.Linear(config.hidden_size, 2)
        self.graft = nn.Linear(config.hidden_size, 2)

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
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = AntiBERTyHeads(config)

        self.init_weights()

        self.num_species = 6
        self.num_chains = 2
        self.num_grafts = 2

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
        return_dict = return_dict if exists(
            return_dict) else self.config.use_return_dict

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
        prediction_scores, species_score, chain_score, graft_score = self.cls(
            sequence_output, pooled_output)

        b = input_ids.shape[0]

        total_loss, masked_lm_loss, species_loss, chain_loss, graft_loss = None, None, None, None, None
        if exists(labels):
            mlm_loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = mlm_loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))

        if exists(species_label):
            species_freqs = torch.bincount(species_label,
                                           minlength=self.num_species)
            species_weights = b / (species_freqs * self.num_species)
            species_loss_fct = nn.CrossEntropyLoss(weight=species_weights)
            species_loss = species_loss_fct(
                species_score.view(-1, self.num_species),
                species_label.view(-1))

        if exists(chain_label):
            chain_freqs = torch.bincount(chain_label,
                                         minlength=self.num_chains)
            species_weights = b / (chain_freqs * self.num_chains)
            chain_loss_fct = nn.CrossEntropyLoss(weight=species_weights)
            chain_loss = chain_loss_fct(chain_score.view(-1, 2),
                                        chain_label.view(-1))

        if exists(graft_label):
            graft_freqs = torch.bincount(graft_label,
                                         minlength=self.num_grafts)
            graft_weights = b / (graft_freqs * self.num_grafts)
            graft_loss_fct = nn.CrossEntropyLoss(weight=graft_weights)
            graft_loss = graft_loss_fct(graft_score.view(-1, 2),
                                        graft_label.view(-1))

        total_loss = \
            masked_lm_loss if exists(masked_lm_loss) else 0 \
            + species_loss if exists(species_loss) else 0 \
            + chain_loss if exists(chain_loss) else 0 \
            + graft_loss if exists(graft_loss) else 0

        if not return_dict:
            output = (prediction_scores, species_score, chain_score,
                      graft_score) + outputs[2:]
            return ((total_loss, ) + output) if exists(total_loss) else output

        return AntiBERTyOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            species_logits=species_score,
            chain_logits=chain_score,
            graft_logits=graft_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
