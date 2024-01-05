from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from transformers.utils import logging, ModelOutput
from transformers.modeling_outputs import MaskedLMOutput
from transformers.activations import gelu
from .tranformer import MolTModel, MolTPreTrainedModel
import math

logger = logging.get_logger(__name__)

class MolTLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class MolTForMaskedLM(MolTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.graph_transformer = MolTModel(config, add_pooling_layer=False)
        self.lm_head = MolTLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pos_embeds: Optional[torch.Tensor] = None,
        pos_embeds_shape: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prop_in_ring: Optional[torch.LongTensor] = None,
        prop_charge: Optional[torch.LongTensor] = None,
        prop_hybridization: Optional[torch.LongTensor] = None,
        prop_chirality: Optional[torch.LongTensor] = None,
        prop_aromatic: Optional[torch.LongTensor] = None,
        prop_conjugated: Optional[torch.LongTensor] = None,
        prop_stereo: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.graph_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pos_embeds=pos_embeds,
            pos_embeds_shape=pos_embeds_shape,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prop_in_ring=prop_in_ring,
            prop_charge=prop_charge,
            prop_hybridization=prop_hybridization,
            prop_chirality=prop_chirality,
            prop_aromatic=prop_aromatic,
            prop_conjugated=prop_conjugated,
            prop_stereo=prop_stereo,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class PropertyPredictionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    confidence: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    components: Optional[torch.FloatTensor] = None
    contributions: Optional[torch.FloatTensor] = None
    epistemic: Optional[torch.FloatTensor] = None
    aleoteric: Optional[torch.FloatTensor] = None



class BayesPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mean_layer = nn.Linear(config.hidden_size, 1, bias=False)
        self.variance_layer = nn.Sequential(
            nn.Linear(config.hidden_size, 1, bias=False), nn.Softplus()
        )

    def forward(self, token_embeddings):
        cls_embedding = token_embeddings[:, [0], :]
        mean = self.mean_layer(cls_embedding).squeeze()
        var = self.variance_layer(cls_embedding).squeeze()
        return Normal(loc=mean, scale=var)


class DERLinear(nn.Module):
    def __init__(self, config):
        super(DERLinear, self).__init__()
        # Linear layer for m (mean)
        self.linear_m = nn.Linear(config.hidden_size, 1)
        # Linear layers for positive outputs (v, alpha, beta)
        self.linear_v = nn.Linear(config.hidden_size, 1)
        self.linear_alpha = nn.Linear(config.hidden_size, 1)
        self.linear_beta = nn.Linear(config.hidden_size, 1)

    def forward(self, token_embeddings):
        x = token_embeddings[:, [0], :]
        m = self.linear_m(x).squeeze()
        v = F.softplus(self.linear_v(x)).squeeze()  # ensure positivity
        alpha = F.softplus(self.linear_alpha(x)).squeeze() + 1  # ensure alpha > 1
        beta = F.softplus(self.linear_beta(x)).squeeze()  # ensure positivity
        return v, alpha, beta, m


def NIG_NLL(y, gamma, v, alpha, beta):
    two_beta_lambda = 2 * beta * (1 + v)

    nll = (
        0.5 * torch.log(math.pi / v)
        - alpha * torch.log(two_beta_lambda)
        + (alpha + 0.5) * torch.log(v * (y - gamma)**2 + two_beta_lambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    return torch.mean(nll)

def der_loss(y_pred, y_true, v, alpha, beta, m):
    loss_pred = NIG_NLL(y_true, gamma=m, v=v, alpha=alpha, beta=beta)
    loss_reg = (y_pred - y_true).abs() * (2 * v + alpha)
    return torch.mean(loss_pred + loss_reg)


class DERMolTProperty(MolTPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.graph_transformer = MolTModel(config, add_pooling_layer=False)
        self.prediction_head = DERLinear(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pos_embeds: Optional[torch.Tensor] = None,
        pos_embeds_shape: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        prop_in_ring: Optional[torch.LongTensor] = None,
        prop_charge: Optional[torch.LongTensor] = None,
        prop_hybridization: Optional[torch.LongTensor] = None,
        prop_chirality: Optional[torch.LongTensor] = None,
        prop_aromatic: Optional[torch.LongTensor] = None,
        prop_conjugated: Optional[torch.LongTensor] = None,
        prop_stereo: Optional[torch.LongTensor] = None,
        reg: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], PropertyPredictionOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.graph_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pos_embeds=pos_embeds,
            pos_embeds_shape=pos_embeds_shape,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
            prop_in_ring=prop_in_ring,
            prop_charge=prop_charge,
            prop_hybridization=prop_hybridization,
            prop_chirality=prop_chirality,
            prop_aromatic=prop_aromatic,
            prop_conjugated=prop_conjugated,
            prop_stereo=prop_stereo,
        )

        # Get only atom embeddings from token_type_ids
        sequence_output = outputs[0]
        v, alpha, beta, m = self.prediction_head(sequence_output)
        mean = m
        variance = (beta * (1 + v)) / (alpha * v)
        prediction_scores = mean

        prediction_loss = None
        if reg is not None:
            reg = reg.to(reg.device)
            prediction_loss = der_loss(mean, reg, v, alpha, beta, m)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((prediction_loss,) + output) if prediction_loss is not None else output
            )

        return PropertyPredictionOutput(
            loss=prediction_loss,
            logits=mean,
            hidden_states=outputs.hidden_states,
            confidence=variance
        )


class SDERLinear(nn.Module):
    def __init__(self, config):
        super(SDERLinear, self).__init__()
        self.linear_gamma = nn.Linear(config.hidden_size, 1)
        self.linear_nu = nn.Linear(config.hidden_size, 1)
        self.linear_beta = nn.Linear(config.hidden_size, 1)

    def forward(self, token_embeddings):
        x = token_embeddings[:, [0], :]
        gamma = self.linear_gamma(x).squeeze()
        nu = F.softplus(self.linear_nu(x)).squeeze()  # ensure positivity
        alpha = nu + 1.0
        beta = F.softplus(self.linear_beta(x)).squeeze()  # ensure positivity
        return gamma, nu, alpha, beta

def loss_sder(y_true, gamma, nu, alpha, beta, coeff):
    error = gamma - y_true
    var = beta / nu
    return torch.mean(torch.log(var) + (1. + coeff * nu) * error**2 / var)

class SDERMolTProperty(MolTPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.graph_transformer = MolTModel(config, add_pooling_layer=False)
        self.prediction_head = SDERLinear(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pos_embeds: Optional[torch.Tensor] = None,
        pos_embeds_shape: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        prop_in_ring: Optional[torch.LongTensor] = None,
        prop_charge: Optional[torch.LongTensor] = None,
        prop_hybridization: Optional[torch.LongTensor] = None,
        prop_chirality: Optional[torch.LongTensor] = None,
        prop_aromatic: Optional[torch.LongTensor] = None,
        prop_conjugated: Optional[torch.LongTensor] = None,
        prop_stereo: Optional[torch.LongTensor] = None,
        reg: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], PropertyPredictionOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.graph_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pos_embeds=pos_embeds,
            pos_embeds_shape=pos_embeds_shape,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
            prop_in_ring=prop_in_ring,
            prop_charge=prop_charge,
            prop_hybridization=prop_hybridization,
            prop_chirality=prop_chirality,
            prop_aromatic=prop_aromatic,
            prop_conjugated=prop_conjugated,
            prop_stereo=prop_stereo,
        )

        # Get only atom embeddings from token_type_ids
        sequence_output = outputs[0]
        gamma, nu, alpha, beta = self.prediction_head(sequence_output)
        prediction_scores = gamma
        epistemic = nu ** (-0.5)
        aleoteric = torch.sqrt((beta*(1 + nu)) / (alpha*nu))

        prediction_loss = None
        if reg is not None:
            reg = reg.to(reg.device)
            prediction_loss = loss_sder(reg, gamma, nu, alpha, beta, coeff=1.0)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((prediction_loss,) + output) if prediction_loss is not None else output
            )

        return PropertyPredictionOutput(
            loss=prediction_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            epistemic=epistemic,
            aleoteric=aleoteric
        )


class MolTProperty(MolTPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.graph_transformer = MolTModel(config, add_pooling_layer=False)
        self.prediction_head = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pos_embeds: Optional[torch.Tensor] = None,
        pos_embeds_shape: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        prop_in_ring: Optional[torch.LongTensor] = None,
        prop_charge: Optional[torch.LongTensor] = None,
        prop_hybridization: Optional[torch.LongTensor] = None,
        prop_chirality: Optional[torch.LongTensor] = None,
        prop_aromatic: Optional[torch.LongTensor] = None,
        prop_conjugated: Optional[torch.LongTensor] = None,
        prop_stereo: Optional[torch.LongTensor] = None,
        reg: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], PropertyPredictionOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.graph_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pos_embeds=pos_embeds,
            pos_embeds_shape=pos_embeds_shape,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
            prop_in_ring=prop_in_ring,
            prop_charge=prop_charge,
            prop_hybridization=prop_hybridization,
            prop_chirality=prop_chirality,
            prop_aromatic=prop_aromatic,
            prop_conjugated=prop_conjugated,
            prop_stereo=prop_stereo,
        )

        # Get only atom embeddings from token_type_ids
        sequence_output = outputs[0]
        prediction_scores = self.prediction_head(sequence_output[:, [0], :]).squeeze()

        prediction_loss = None
        if reg is not None:
            reg = reg.to(reg.device)
            prediction_loss = F.mse_loss(prediction_scores, reg)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((prediction_loss,) + output) if prediction_loss is not None else output
            )

        return PropertyPredictionOutput(
            loss=prediction_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
        )
