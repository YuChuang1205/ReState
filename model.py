import copy
import math
import os
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions, ModelOutput, Seq2SeqLMOutput,
    Seq2SeqModelOutput)
# from transformers.models.t5.modeling_t5 import (__HEAD_MASK_WARNING_MSG,
#                                                 T5Block, T5LayerNorm, T5Stack)
try:
    from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG
except ImportError:
    __HEAD_MASK_WARNING_MSG = (
        "The input argument `head_mask` was separated into two arguments: "
        "`head_mask` and `decoder_head_mask`. Please use `decoder_head_mask` "
        "for decoder self-attention heads."
    )

from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm, T5Stack
# from transformers.utils.model_parallel_utils import (assert_device_map,
#                                                      get_device_map)
try:
    from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
except Exception:
    def get_device_map(n_layers, devices):
        devices = list(devices)
        if len(devices) == 0:
            return {0: list(range(n_layers))}

        layers = list(range(n_layers))
        per_device = n_layers // len(devices)
        remainder = n_layers % len(devices)
        device_map = {}
        start = 0
        for i, device in enumerate(devices):
            count = per_device + (1 if i < remainder else 0)
            if count > 0:
                device_map[device] = layers[start:start + count]
            start += count
        return device_map

    def assert_device_map(device_map, num_layers):
        covered = []
        for layers in device_map.values():
            covered.extend(list(layers))
        covered = sorted(covered)
        expected = list(range(num_layers))
        if covered != expected:
            raise ValueError(
                f"Invalid device_map. Expected layers {expected}, but got {covered}."
            )

class JointEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None, patch_size=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.patch_num, self.patch_dim = patch_size
        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        image_ids=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        image_embedding = self.image_dense(image_ids)

        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding)
        
        merge = torch.cat([hidden_states, image_att], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        hidden_states = (1 - gate) * hidden_states + gate * image_att

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )





def contrastive_loss_topk_extreme(
    pred_embed, pos_label_list, neg_label_list, cls_linear,
    margin=0.1, labels=None, top_k=5
):
    pred_embed = F.normalize(pred_embed, dim=-1)
    pos_label_list = pos_label_list.permute(1, 0, 2)
    neg_label_list = neg_label_list.permute(1, 0, 2)

    sim_pos_all = torch.cat([
        torch.cosine_similarity(pred_embed, F.normalize(cls_linear(pl), dim=-1), dim=-1).unsqueeze(0)
        for pl in pos_label_list
    ], dim=0)

    sim_neg_all = torch.cat([
        torch.cosine_similarity(pred_embed, F.normalize(cls_linear(nl), dim=-1), dim=-1).unsqueeze(0)
        for nl in neg_label_list
    ], dim=0)

    k_pos = min(top_k, sim_pos_all.size(0))
    lowk_pos_sim, _ = sim_pos_all.topk(k_pos, dim=0, largest=False)
    avg_lowk_pos_sim = lowk_pos_sim.mean(dim=0)

    k_neg = min(top_k, sim_neg_all.size(0))
    highk_neg_sim, _ = sim_neg_all.topk(k_neg, dim=0, largest=True)
    avg_highk_neg_sim = highk_neg_sim.mean(dim=0)

    loss = F.relu(avg_highk_neg_sim + margin - avg_lowk_pos_sim)
    if labels is not None:
        mask = (labels != -100).float()
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
    else:
        loss = loss.mean()
    return loss



class Cls_linear(nn.Module):
    def __init__(self, i_dim, embeddings):
        super().__init__()
        self.embeddings = embeddings
        self.linear = nn.Linear(i_dim, i_dim)

    def forward(self, x):
        y = self.embeddings(x)
        y = self.linear(y)
        #y = torch.mean(y, dim = 1)
        return y


from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class Seq2SeqLMOutputWithContrastive(Seq2SeqLMOutput):
    ce_loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None


class T5ForMultimodalGeneration(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size):
        super().__init__(config)
        self.hardsamplemine = False
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # self.encoder = T5Stack(encoder_config, self.shared)
        self.encoder = JointEncoder(encoder_config, self.shared, patch_size)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.cls_linear = Cls_linear(config.d_model, self.shared)
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_hardsamplemine(self, hardsamplemine, margin=0.2, alpha=0.1, topk=5):
        self.hardsamplemine = hardsamplemine
        self.alpha = alpha
        self.margin = margin
        self.topk = topk

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_ids=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_labels = None,
        neg_labels = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                image_ids=image_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            ce_loss = CrossEntropyLoss(ignore_index=-100)(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            if self.hardsamplemine and pos_labels is not None and neg_labels is not None:
                pred_token_ids = lm_logits.argmax(dim=-1)  # [B, T]
                pred_embed = self.cls_linear(pred_token_ids)  # [B, T, D]
                contrastive_loss = contrastive_loss_topk_extreme(pred_embed, pos_labels, neg_labels, self.cls_linear, margin=self.margin, labels=labels, top_k=self.topk)
                loss = ce_loss + self.alpha * contrastive_loss
            else:
                loss = ce_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output


        return Seq2SeqLMOutputWithContrastive(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            ce_loss=ce_loss if labels is not None else None,
            contrastive_loss=self.alpha*contrastive_loss if (
                    self.hardsamplemine and pos_labels is not None and neg_labels is not None) else None,
        )


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
    # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if "image_ids" in kwargs:
            output["image_ids"] = kwargs['image_ids']

        return output
    
    def test_step(self, tokenizer, batch, **kwargs):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        image_ids = batch['image_ids'].to(device)

        output = self.generate(
            input_ids=input_ids,
            image_ids=image_ids,
            **kwargs
        )

        generated_sents = tokenizer.batch_decode(output, skip_special_tokens=True)
        targets = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        result = {}
        result['preds'] = generated_sents
        result['targets'] = targets

        return result



# ============================================================
# Qwen2.5-VL / Qwen3-VL wrappers for keeping your Seq2SeqTrainer
# + evaluation pipeline unchanged.
# ============================================================

from transformers.modeling_outputs import CausalLMOutputWithPast


def _is_qwen35(model_name: str) -> bool:
    """Qwen3.5 is multimodal but official repo names can be like Qwen/Qwen3.5-9B without "VL"."""
    n = (model_name or "").lower()
    return "qwen3.5" in n or "qwen3_5" in n or "qwen35" in n

def _is_qwen3(model_name: str) -> bool:
    n = (model_name or "").lower()
    return ("qwen3" in n and "vl" in n) and not _is_qwen35(model_name)

def _is_qwen25(model_name: str) -> bool:
    n = (model_name or "").lower()
    return ("qwen2.5" in n or "qwen2_5" in n) and "vl" in n

def _from_pretrained_compat(model_cls, model_name: str, torch_dtype=None, device_map=None, attn_implementation="sdpa"):
    """Small compatibility wrapper across transformers versions/classes."""
    kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return model_cls.from_pretrained(model_name, **kwargs)
    except TypeError:
        # Some older/newer classes may not expose attn_implementation in from_pretrained.
        kwargs.pop("attn_implementation", None)
        return model_cls.from_pretrained(model_name, **kwargs)

def load_qwen_vl_base(model_name: str, torch_dtype=None, device_map=None, attn_implementation="sdpa"):
    """
    Train:  device_map=None  (DDP/FSDP/ZeRO)
    Infer:  device_map="auto"

    Supports Qwen2.5-VL, Qwen3-VL, and Qwen3.5 multimodal checkpoints
    such as Qwen/Qwen3.5-9B whose repo name may not contain "VL".
    """
    # Qwen3.5 must be checked before Qwen3-VL because some repos may contain both "qwen3.5" and "vl".
    if _is_qwen35(model_name):
        try:
            from transformers import Qwen3_5ForConditionalGeneration
            return _from_pretrained_compat(
                Qwen3_5ForConditionalGeneration,
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
        except Exception as e1:
            try:
                # Compatibility with environments that expose a VL-suffixed class name.
                from transformers import Qwen3_5VLForConditionalGeneration
                return _from_pretrained_compat(
                    Qwen3_5VLForConditionalGeneration,
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                )
            except Exception as e2:
                try:
                    from transformers import AutoModelForImageTextToText
                    return _from_pretrained_compat(
                        AutoModelForImageTextToText,
                        model_name,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        attn_implementation=attn_implementation,
                    )
                except Exception:
                    raise RuntimeError(
                        f"Failed to load Qwen3.5 checkpoint: {model_name}. "
                        f"Please install a transformers version that supports Qwen3.5. "
                        f"Qwen3_5ForConditionalGeneration error: {repr(e1)}; "
                        f"Qwen3_5VLForConditionalGeneration error: {repr(e2)}"
                    )

    if _is_qwen25(model_name):
        from transformers import Qwen2_5_VLForConditionalGeneration
        return _from_pretrained_compat(
            Qwen2_5_VLForConditionalGeneration,
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    if _is_qwen3(model_name):
        from transformers import Qwen3VLForConditionalGeneration
        return _from_pretrained_compat(
            Qwen3VLForConditionalGeneration,
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    # fallback (other VLMs): try the most compatible auto classes
    try:
        from transformers import AutoModelForImageTextToText
        return _from_pretrained_compat(
            AutoModelForImageTextToText,
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
    except Exception:
        pass
    try:
        from transformers import AutoModelForVision2Seq
        return _from_pretrained_compat(
            AutoModelForVision2Seq,
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
    except Exception:
        # last resort for text-only or unsupported multimodal auto-mapping
        from transformers import AutoModelForCausalLM
        return _from_pretrained_compat(
            AutoModelForCausalLM,
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )


class QwenVLForScienceQA(nn.Module):
    """
    Wrapper:
      - forward() accepts prompt-only input_ids + target-only labels (like seq2seq).
      - internally concatenates prompt + labels to compute causal LM loss.
      - generate() trims prompt tokens (so metrics decode sees only new tokens).
      - keeps optional hard negative mining (pos_labels/neg_labels) via existing contrastive loss.
    """
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.model = base_model
        
        ##add
        self.config = getattr(self.model, "config", None)
        from transformers.generation import GenerationConfig
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.generation_config = self.model.generation_config
        elif self.config is not None:
            self.generation_config = GenerationConfig.from_model_config(self.config)
        else:
            self.generation_config = GenerationConfig()


        self.tokenizer = tokenizer
        self.hardsamplemine = False
        self.alpha = 0.1
        self.margin = 0.2
        self.topk = 5
        # CLS linear uses underlying embedding layer
        # hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None)
        # self.cls_linear = Cls_linear(hidden, self.model.get_input_embeddings())
        emb = self.model.get_input_embeddings()
        hidden = getattr(emb, "embedding_dim", None)
        if hidden is None:
            hidden = emb.weight.shape[-1]  # vocab x hidden
        hidden = int(hidden)
        assert isinstance(hidden, int) and hidden > 0, f"Invalid hidden size: {hidden}"
        self.cls_linear = Cls_linear(hidden, emb)
        print("config.hidden_size:", getattr(self.model.config, "hidden_size", None))
        print("config.d_model:", getattr(self.model.config, "d_model", None))
        emb = self.model.get_input_embeddings()
        print("emb.weight.shape:", emb.weight.shape, "embedding_dim:", getattr(emb, "embedding_dim", None))


    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False, **kwargs):
        """
        Compatibility wrapper for newer HuggingFace Trainer versions.

        Some transformers versions call:
            model.num_parameters(exclude_embeddings=True)
        when estimating FLOPs. This value is only used for FLOPs logging,
        not for forward/loss/optimization.
        """
        if hasattr(self.model, "num_parameters"):
            try:
                return self.model.num_parameters(
                    only_trainable=only_trainable,
                    exclude_embeddings=exclude_embeddings,
                )
            except TypeError:
                try:
                    return self.model.num_parameters(only_trainable=only_trainable)
                except TypeError:
                    pass

        params = list(self.parameters())

        if only_trainable:
            params = [p for p in params if p.requires_grad]

        if exclude_embeddings:
            embedding_param_ids = set()
            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    for p in module.parameters(recurse=False):
                        embedding_param_ids.add(id(p))
            params = [p for p in params if id(p) not in embedding_param_ids]

        return sum(p.numel() for p in params)

    def set_hardsamplemine(self, hardsamplemine, margin=0.2, alpha=0.1, topk=5):
        self.hardsamplemine = hardsamplemine
        self.alpha = alpha
        self.margin = margin
        self.topk = topk

    def save_pretrained(self, save_directory, **kwargs):
        # Delegate to base model to keep HF checkpoint format
        return self.model.save_pretrained(save_directory, **kwargs)

    def _seq_mean_pool(self, token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # [1, L]

        emb = self.model.get_input_embeddings()(token_ids)

        # mask pad
        mask = (token_ids != pad_id).float().unsqueeze(-1)   # [B, L, 1]
        emb = emb * mask
        denom = mask.sum(dim=1).clamp_min(1.0)               # [B, 1]
        pooled = emb.sum(dim=1) / denom                      # [B, H]
        return pooled

    # ---------------------------------------------------------
    # HF Trainer hooks: gradient checkpointing + cache control
    # ---------------------------------------------------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        HF Trainer may call `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)`.
        Some transformers versions/models do NOT accept kwargs (e.g., use_reentrant).
        We make this wrapper robust across versions.
        """
        m = getattr(self, "model", None)
        if m is None or not hasattr(m, "gradient_checkpointing_enable"):
            return self

        if gradient_checkpointing_kwargs:
            try:
                m.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
                return self
            except TypeError:
                pass

        m.gradient_checkpointing_enable()
        return self


    def gradient_checkpointing_disable(self):
        m = getattr(self, "model", None)
        if m is not None and hasattr(m, "gradient_checkpointing_disable"):
            m.gradient_checkpointing_disable()
        return self

    @property
    def is_gradient_checkpointing(self):
        m = getattr(self, "model", None)
        if m is not None and hasattr(m, "is_gradient_checkpointing"):
            return m.is_gradient_checkpointing
        cfg = getattr(m, "config", None)
        return bool(getattr(cfg, "gradient_checkpointing", False)) if cfg is not None else False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pos_labels=None,
        neg_labels=None,
        **kwargs
    ):
        """
        input_ids: [B, Lp] prompt-only
        labels:    [B, Lt] target-only (padded with pad_token_id)
        """
        if input_ids is None:
            raise ValueError("input_ids is required for QwenVLForScienceQA")
        device = input_ids.device
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Build full sequence for loss: [prompt | target]
        if labels is not None:
            emb_dev = self.model.get_input_embeddings().weight.device
            if input_ids.device != emb_dev:
                input_ids = input_ids.to(emb_dev)
            if attention_mask is not None and attention_mask.device != emb_dev:
                attention_mask = attention_mask.to(emb_dev)
            if labels is not None and torch.is_tensor(labels) and labels.device != emb_dev:
                labels = labels.to(emb_dev)

            for k, v in list(kwargs.items()):
                if torch.is_tensor(v) and v.device != emb_dev:
                    kwargs[k] = v.to(emb_dev)

            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.long, device=emb_dev)
            else:
                labels = labels.to(emb_dev)

            # Full input = prompt + target_ids (pad kept)
            full_input_ids = torch.cat([input_ids, labels], dim=1)

            # Full attention = prompt mask + (labels != pad)
            if attention_mask is None:
                attention_mask = (input_ids != pad_id).long()
            full_attention = torch.cat([attention_mask, (labels != pad_id).long()], dim=1)

            # Full labels: ignore prompt tokens, ignore padding tokens in target
            prompt_ignore = torch.full_like(input_ids, -100)
            full_labels = torch.cat([prompt_ignore, labels.clone()], dim=1)
            full_labels[full_labels == pad_id] = -100
            for _tk in [
                "input_token_type",
                "input_token_type_ids",
                "mm_token_type_ids",
            ]:
                _tti = kwargs.get(_tk, None)
                if _tti is None or not torch.is_tensor(_tti):
                    continue
                if _tti.device != emb_dev:
                    _tti = _tti.to(emb_dev)

                if _tti.dim() >= 2 and _tti.size(0) == input_ids.size(0):
                    if _tti.size(1) == input_ids.size(1):
                        target_shape = list(_tti.shape)
                        target_shape[1] = labels.size(1)
                        target_tti = torch.zeros(
                            target_shape,
                            dtype=_tti.dtype,
                            device=_tti.device,
                        )
                        kwargs[_tk] = torch.cat([_tti, target_tti], dim=1)

                    elif _tti.size(1) == full_input_ids.size(1):
                        kwargs[_tk] = _tti

                    else:
                        raise ValueError(
                            f"{_tk} length mismatch: got {_tti.size(1)}, "
                            f"prompt_len={input_ids.size(1)}, full_len={full_input_ids.size(1)}"
                        )

            _pos = kwargs.get("position_ids", None)
            if torch.is_tensor(_pos) and _pos.dim() >= 2 and _pos.size(-1) != full_input_ids.size(1):
                kwargs.pop("position_ids", None)

            pv = kwargs.get("pixel_values", None)
            g  = kwargs.get("image_grid_thw", None)

            if pv is not None:
                assert torch.is_tensor(pv), f"pixel_values must be a tensor, got {type(pv)}"
                assert pv.dim() in (2, 3, 4), f"unexpected pixel_values shape: {pv.shape}"

            if g is not None:
                assert torch.is_tensor(g), f"image_grid_thw must be a tensor, got {type(g)}"
                assert g.dim() == 2 and g.size(-1) == 3, f"image_grid_thw should be [N,3], got {g.shape}"

            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=full_attention,
                labels=full_labels,
                **kwargs,
                return_dict=True
            )
            loss = outputs.loss

            ce_loss = outputs.loss
            contrastive_loss = None
            logits_full = outputs.logits

            # IMPORTANT for compatibility with your existing evaluation code:
            # - When predict_with_generate=False, Seq2SeqTrainer will use logits to decode.
            # - Your code expects logits to align with label length (target_len).
            # Therefore we slice logits to ONLY the target part (last Lt tokens).
            target_len = labels.size(1)
            logits = logits_full[:, -target_len:, :]

            # Optional contrastive hard mining loss (kept consistent with your original implementation)
            if self.hardsamplemine and pos_labels is not None and neg_labels is not None:
                if not torch.is_tensor(pos_labels):
                    pos_labels = torch.tensor(pos_labels, dtype=torch.long, device=device)
                else:
                    pos_labels = pos_labels.to(device)
                if not torch.is_tensor(neg_labels):
                    neg_labels = torch.tensor(neg_labels, dtype=torch.long, device=device)
                else:
                    neg_labels = neg_labels.to(device)

                # Use full logits for mining so the model can mine from the whole generated sequence
                pred_token_ids = logits_full.argmax(dim=-1)  # [B, Lp+Lt]

                pred_embed = self._seq_mean_pool(pred_token_ids, pad_id=pad_id)  # [B, H]

                def aligned_cls(token_ids: torch.Tensor) -> torch.Tensor:
                    return self._seq_mean_pool(token_ids.to(device), pad_id=pad_id)  # [B, H]

                contrastive_loss = contrastive_loss_topk_extreme(
                    pred_embed,        # [B, H]
                    pos_labels,
                    neg_labels,
                    aligned_cls,       # returns [B, H]
                    margin=self.margin,
                    labels=full_labels,
                    top_k=self.topk
                )
                loss = loss + self.alpha * contrastive_loss

            out = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            out.ce_loss = ce_loss
            out.contrastive_loss = contrastive_loss
            return out

        # No labels: pure forward (used in generation/eval fallback).
        # Keep auxiliary loss fields defined for all call paths.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
            return_dict=True
        )
        outputs.ce_loss = None
        outputs.contrastive_loss = None
        return outputs

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        Trim the prompt tokens after generation so your compute_metrics decodes
        only the newly generated assistant tokens.
        (As shown in official HF docs for Qwen2.5-VL / Qwen3-VL.)
        """
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        if input_ids is None:
            raise ValueError("generate() requires input_ids")

        # Some processors add token_type_ids; remove for safety
        kwargs.pop("token_type_ids", None)
        kwargs.pop("pos_labels", None)
        kwargs.pop("neg_labels", None)
        kwargs.pop("labels", None)
        # if kwargs.get("max_new_tokens", None) is not None:
        #     kwargs.pop("max_length", None)
        # ===== HF high/low-version generation compatibility =====
        # Goal:
        # 1. Avoid warning: generation_config + max_new_tokens
        # 2. Avoid warning: max_new_tokens + generation_config.max_length=20
        # 3. Preserve original generation_config as much as possible
        max_new_tokens = kwargs.pop("max_new_tokens", None)

        if max_new_tokens is not None:
            kwargs.pop("max_length", None)

            input_ids_for_len = kwargs.get("input_ids", None)
            if input_ids_for_len is None and len(args) > 0 and torch.is_tensor(args[0]):
                input_ids_for_len = args[0]

            if input_ids_for_len is not None and torch.is_tensor(input_ids_for_len):
                effective_max_length = int(input_ids_for_len.shape[-1]) + int(max_new_tokens)

                gen_cfg = kwargs.pop("generation_config", None)

                if gen_cfg is None and hasattr(self.model, "generation_config"):
                    gen_cfg = copy.deepcopy(self.model.generation_config)
                elif gen_cfg is not None:
                    gen_cfg = copy.deepcopy(gen_cfg)

                if gen_cfg is not None:
                    gen_cfg.max_length = effective_max_length
                    if hasattr(gen_cfg, "max_new_tokens"):
                        gen_cfg.max_new_tokens = None
                    kwargs["generation_config"] = gen_cfg
                else:
                    kwargs["max_length"] = effective_max_length
            else:
                kwargs["max_new_tokens"] = int(max_new_tokens)

        gen_cfg = kwargs.get("generation_config", None)
        if gen_cfg is None and hasattr(self, "generation_config"):
            gen_cfg = copy.deepcopy(self.generation_config)

        if gen_cfg is not None:
            do_sample = kwargs.get("do_sample", getattr(gen_cfg, "do_sample", False))
            if not do_sample:
                if hasattr(gen_cfg, "temperature"):
                    gen_cfg.temperature = 1.0
                if hasattr(gen_cfg, "top_p"):
                    gen_cfg.top_p = 1.0
                if hasattr(gen_cfg, "typical_p"):
                    gen_cfg.typical_p = 1.0
                if hasattr(gen_cfg, "min_p"):
                    gen_cfg.min_p = None
                if hasattr(gen_cfg, "epsilon_cutoff"):
                    gen_cfg.epsilon_cutoff = 0.0
                if hasattr(gen_cfg, "eta_cutoff"):
                    gen_cfg.eta_cutoff = 0.0

            kwargs["generation_config"] = gen_cfg

        out = self.model.generate(*args, **kwargs)

        # out: [B, Lp+Lg]
        if torch.is_tensor(out) and torch.is_tensor(input_ids):
            trimmed = []
            max_len = 0
            for in_ids, out_ids in zip(input_ids, out):
                t = out_ids[len(in_ids):]
                trimmed.append(t)
                max_len = max(max_len, t.numel())
            # pad to same length
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded = []
            for t in trimmed:
                if t.numel() < max_len:
                    t = torch.cat([t, torch.full((max_len - t.numel(),), pad_id, dtype=t.dtype, device=t.device)], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0)
        return out
