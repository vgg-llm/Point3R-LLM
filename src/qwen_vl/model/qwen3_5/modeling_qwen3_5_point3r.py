import torch
import torch.nn as nn
from typing import Optional, Any, Tuple, List, Dict, Union

from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5CausalLMOutputWithPast
from .configuration_qwen3_5 import Qwen3_5Config
from ..point3r.point3r import Point3R, Point3RConfig
from ..feature_fusion import FeatureFusionModule, FeatureFusionConfig, FeatureProjector
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache


class Qwen3_5ForConditionalGenerationWithPoint3R(Qwen3_5ForConditionalGeneration):
    """
    Qwen3.5 model with Point3R 3D memory support.

    This class extends Qwen3_5ForConditionalGeneration to support pointer memory embeddings
    from Point3R, enabling 3D scene understanding capabilities.

    Key differences from Qwen3VLForConditionalGenerationWithPoint3R:
    - No DeepStack (Qwen3.5 does not have deepstack_visual_indexes)
    - Uses mm_token_type_ids for position ID computation (Qwen3.5-specific)
    - Vision model returns BaseModelOutputWithPooling instead of tuple with deepstack list
    - Uses is_first_iteration flag in prepare_inputs_for_generation (not cache_position)
    - Calls self.model.language_model() directly after embedding injection
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)

        # Pointer token ID obtained from tokenizer (set after tokenizer.add_special_tokens)
        self.pointer_token_id = getattr(config, 'pointer_token_id', None)

        # Initialize Point3R model if needed (for on-the-fly feature extraction)
        if getattr(config, 'use_pointer_memory', False) and not getattr(config, 'use_preprocessed_input', False):
            self._init_point3r_memory(config)

        # Initialize memory feature fusion modules if enabled
        if getattr(config, 'merge_memory_feat', False):
            self._init_memory_fusion(config)
        elif getattr(config, 'tune_feature_projector', False):
            self._init_feature_projector(config)

        # Initialize pointer position encoder if enabled
        if getattr(config, 'use_pointer_position_encoding', False):
            self._init_pointer_position_encoder(config)

        # Initialize RoPE3D-Continuous encoder if enabled
        if getattr(config, 'rope_mode', 'none') == 'continuous':
            self._init_rope3d_continuous(config)

        self.post_init()

        # Apply custom weight initialization AFTER post_init() to prevent overwriting
        self._init_custom_weights()

    def _init_point3r_memory(self, config):
        """Initialize Point3R model for on-the-fly 3D feature extraction."""
        inf = float('inf')
        point3r_config = Point3RConfig(
            freeze='encoder',
            pos_embed='RoPE100',
            pos_embed_3d='RoPE3D100',
            pose_head=True,
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='dpt',
            output_mode='pts3d+pose',
            depth_mode=('exp', -inf, inf),
            conf_mode=('exp', 1, inf),
            pose_mode=('exp', -inf, inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            landscape_only=False
        )
        self.point3r_model = Point3R(point3r_config)
        self.point3r_model.eval()

    def _init_feature_projector(self, config):
        """Initialize feature projector for pointer embeddings (without memory fusion).

        This projector maintains dimensions (no size change) and is used when
        memory fusion is disabled but feature projection is enabled.
        """
        hidden_size = config.text_config.hidden_size

        self.feature_projector = FeatureProjector(
            input_dim=hidden_size,
            output_dim=hidden_size,  # Same dimension - no size change
            hidden_dim=getattr(config, "memory_merger_hidden_dim", 4096),
        )

    def _init_pointer_position_encoder(self, config):
        """Initialize learnable position encoder for pointer memory tokens.

        This encoder projects continuous 3D coordinates (xyz) to embedding space
        and adds the position encoding to pointer embeddings before LLM injection.
        """
        from ..feature_fusion import PointerPositionEncoder

        self.pointer_position_encoder = PointerPositionEncoder(
            coord_dim=3,  # xyz coordinates
            hidden_dim=getattr(config, 'pointer_pos_hidden_dim', 256),
            out_dim=config.text_config.hidden_size,
        )

    def _init_rope3d_continuous(self, config):
        """Initialize RoPE3DContinuous encoder with output projector.

        RoPE3DContinuous preserves input dimension - it splits the embedding into 3 parts
        for x, y, z encoding and concatenates them back. No input projection needed.
        """
        from ..croco.pos_embed_con import RoPE3DContinuous

        hidden_size = config.text_config.hidden_size

        self.rope3d_continuous = RoPE3DContinuous(freq=100.0, F0=1.0)

        # Output projector for learnable residual adaptation
        # Note: Custom weight init (small weights) is done in _init_custom_weights() AFTER post_init()
        self.rope3d_output_projector = nn.Linear(hidden_size, hidden_size)

    def _init_custom_weights(self):
        """Apply custom weight initialization for residual learning modules.

        Must be called AFTER post_init() to prevent overwriting by _init_weights().
        """
        with torch.no_grad():
            # RoPE3D continuous projector - small weights for residual learning
            if hasattr(self, 'rope3d_output_projector'):
                self.rope3d_output_projector.weight.data.mul_(0.001)
                self.rope3d_output_projector.bias.data.zero_()

            # Memory feature projector - small weights for residual learning
            if hasattr(self, 'memory_feature_projector'):
                for module in self.memory_feature_projector.modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.mul_(0.001)
                        if module.bias is not None:
                            module.bias.data.zero_()

            # Memory feature fusion module - custom initialization based on fusion method
            if hasattr(self, 'memory_feature_fusion'):
                if hasattr(self.memory_feature_fusion, 'fusion_method'):
                    if self.memory_feature_fusion.fusion_method == "weighted":
                        if hasattr(self.memory_feature_fusion, 'weight_2d'):
                            self.memory_feature_fusion.weight_2d.fill_(0.5)
                        if hasattr(self.memory_feature_fusion, 'weight_3d'):
                            self.memory_feature_fusion.weight_3d.fill_(0.5)
                    elif self.memory_feature_fusion.fusion_method in ["cross_attention", "self_attention"]:
                        for module in self.memory_feature_fusion.modules():
                            if isinstance(module, nn.Linear):
                                module.weight.data.mul_(0.01)
                                if module.bias is not None:
                                    module.bias.data.zero_()

    def _apply_continuous_rope(self, pointer_embeds, positions):
        """Apply RoPE3DContinuous and add to embeddings (additive fusion).

        Args:
            pointer_embeds: Pointer embeddings (num_pointers, hidden_size)
            positions: 3D positions for each pointer (num_pointers, 3) as [h, w, d]

        Returns:
            Pointer embeddings with continuous RoPE position encoding added
        """
        # Reshape for RoPE3DContinuous: expects (B, heads, ntokens, D)
        embeds = pointer_embeds.unsqueeze(0).unsqueeze(0)  # (1, 1, N, hidden_size)
        pos = positions.unsqueeze(0)  # (1, N, 3)

        encoded = self.rope3d_continuous(embeds, pos)  # (1, 1, N, hidden_size)

        rope_features = self.rope3d_output_projector(encoded)
        rope_features = rope_features.squeeze(0).squeeze(0)  # (N, hidden_size)

        return pointer_embeds + rope_features  # Additive fusion

    def _init_memory_fusion(self, config):
        """Initialize memory feature fusion modules for Point3R memory_feat integration."""
        memory_dim = 768  # Point3R dec_embed_dim
        output_dim = config.text_config.hidden_size

        self.memory_feature_projector = FeatureProjector(
            input_dim=memory_dim,
            output_dim=output_dim,
            hidden_dim=getattr(config, "memory_merger_hidden_dim", 4096),
        )

        fusion_config = FeatureFusionConfig(
            fusion_method=getattr(config, "memory_fusion_method", "add"),
            hidden_size=output_dim,
            num_heads=getattr(config, "memory_fusion_attention_heads", 8),
            dropout=getattr(config, "memory_fusion_dropout", 0.1),
            num_layers=getattr(config, "memory_fusion_num_layers", 1)
        )
        self.memory_feature_fusion = FeatureFusionModule(fusion_config)

    def _process_memory_features(self, pointer_memory_embeds, memory_feat):
        """Process Point3R memory features and fuse with pointer embeddings.

        Args:
            pointer_memory_embeds: Tensor of shape (num_pointers, text_hidden_size)
            memory_feat: Tensor of shape (num_pointers, 768)

        Returns:
            Tensor of shape (num_pointers, text_hidden_size)
        """
        if memory_feat is None:
            return pointer_memory_embeds

        memory_feat = memory_feat.to(pointer_memory_embeds.dtype)

        num_pointers, memory_dim = memory_feat.shape
        memory_feat_spatial = memory_feat.view(num_pointers, 1, 1, memory_dim)

        merged_memory = self.memory_feature_projector(memory_feat_spatial)
        merged_memory = merged_memory.view(num_pointers, -1)

        fused_embeds = self.memory_feature_fusion(pointer_memory_embeds, merged_memory)

        return fused_embeds

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model with optional Point3R checkpoint."""
        point3r_model_path = kwargs.pop("point3r_model_path", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if point3r_model_path:
            model.point3r_model = Point3R.from_pretrained(point3r_model_path)
            model.point3r_model.eval()
        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        pointer_memory_embeds: Optional[torch.Tensor] = None,
        pointer_positions: Optional[torch.Tensor] = None,
        memory_feat: Optional[torch.FloatTensor] = None,
        pointer_timestamps: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple | Qwen3_5CausalLMOutputWithPast:
        """
        Forward pass with pointer memory support.

        Args:
            pointer_memory_embeds: Point3R memory embeddings (num_pointers, hidden_size)
            pointer_positions: 3D positions for each pointer (num_pointers, 3) as [h, w, d]
            memory_feat: Point3R internal decoder features (num_pointers, 768) for fusion
            pointer_timestamps: Per-token frame indices (num_pointers,) for timestamp-grouped RoPE
            mm_token_type_ids: Token type IDs (batch, seq_len): 0=text, 1=image, 2=video
            ... (other args same as parent)

        Note: No deepstack_pointer_embeds — Qwen3.5 does not have DeepStack.
        """
        # Step 1: Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Step 2: Handle image embeddings
        if pixel_values is not None:
            image_outputs = self.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Step 3: Handle video embeddings
        if pixel_values_videos is not None:
            video_outputs = self.model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Step 4: Handle pointer memory embeddings
        if pointer_memory_embeds is not None:
            pointer_memory_embeds = pointer_memory_embeds.type(self.model.visual.dtype)
            n_pointer_tokens = (input_ids == self.pointer_token_id).sum().item()
            n_pointer_features = pointer_memory_embeds.shape[0]

            if n_pointer_tokens != n_pointer_features:
                raise ValueError(
                    f"Pointer memory features and pointer tokens do not match: "
                    f"tokens: {n_pointer_tokens}, features {n_pointer_features}"
                )

            # Fuse with memory_feat if enabled (before masked_scatter)
            if getattr(self.config, 'merge_memory_feat', False) and memory_feat is not None:
                pointer_memory_embeds = self._process_memory_features(pointer_memory_embeds, memory_feat)
            # Simple projection without memory fusion
            elif hasattr(self, 'feature_projector'):
                pointer_memory_embeds = self.feature_projector(pointer_memory_embeds)

            # Apply position encoding if enabled and positions available
            if hasattr(self, 'pointer_position_encoder') and pointer_positions is not None:
                pointer_memory_embeds = self.pointer_position_encoder(
                    pointer_memory_embeds, pointer_positions
                )

            # Apply RoPE3D-Continuous if enabled
            if hasattr(self, 'rope3d_continuous') and pointer_positions is not None:
                pointer_memory_embeds = self._apply_continuous_rope(
                    pointer_memory_embeds, pointer_positions
                )

            # Scatter pointer embeddings into inputs_embeds
            pointer_mask = (input_ids == self.pointer_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            pointer_mask = pointer_mask.to(inputs_embeds.device)
            pointer_memory_embeds = pointer_memory_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(pointer_mask, pointer_memory_embeds)

        # Step 5: Compute position IDs
        # Qwen3.5-specific: use compute_3d_position_ids with mm_token_type_ids.
        # Pointer tokens in video format are wrapped in vision_start/end, so mm_token_type_ids
        # may label them as type 2 (video). We mask them as type 0 (text) so that
        # get_rope_index doesn't try to consume video_grid_thw entries for them.
        if position_ids is None:
            eff_mm_token_type_ids = mm_token_type_ids
            if mm_token_type_ids is not None and self.pointer_token_id is not None and input_ids is not None:
                ptr_mask = (input_ids == self.pointer_token_id)
                if ptr_mask.any():
                    eff_mm_token_type_ids = mm_token_type_ids.clone()
                    eff_mm_token_type_ids[ptr_mask] = 0  # treat pointer tokens as text for RoPE

            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=eff_mm_token_type_ids,
            )

        # Step 6: Forward through language model directly
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Compute logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.model.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        pointer_memory_embeds=None,
        pointer_positions=None,
        memory_feat=None,
        pointer_timestamps=None,
        **kwargs,
    ):
        """Prepare inputs for generation with pointer memory support."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # Add pointer-specific inputs
        model_inputs["pointer_memory_embeds"] = pointer_memory_embeds
        model_inputs["pointer_positions"] = pointer_positions
        model_inputs["memory_feat"] = memory_feat
        model_inputs["pointer_timestamps"] = pointer_timestamps

        if not is_first_iteration and use_cache:
            # After the prefill phase, pointer inputs should not be forwarded
            # as they have already been processed and are stored in the KV cache
            model_inputs["pointer_memory_embeds"] = None
            model_inputs["pointer_positions"] = None
            model_inputs["memory_feat"] = None
            model_inputs["pointer_timestamps"] = None

        return model_inputs

    def _get_pointer_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """
        Get the number of pointer tokens for each sample in the batch.

        Args:
            input_ids: Input token IDs (batch_size, seq_length)

        Returns:
            pointer_nums: Number of pointer tokens per sample (batch_size,)
        """
        pointer_token_id = self.pointer_token_id
        if pointer_token_id is None:
            return torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        pointer_mask = input_ids == pointer_token_id
        pointer_nums = torch.sum(pointer_mask, dim=1)

        return pointer_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expand inputs for beam search with pointer memory support."""
        if expand_size == 1:
            return input_ids, model_kwargs

        # Add pointer keys to visual keys (Qwen3.5 uses mm_token_type_ids, no second_per_grid_ts)
        visual_keys = [
            "pixel_values", "image_grid_thw", "pixel_values_videos",
            "video_grid_thw", "mm_token_type_ids",
            "pointer_memory_embeds", "pointer_positions", "memory_feat", "pointer_timestamps",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )
            pointer_nums = self._get_pointer_nums(input_ids)

            if video_grid_thw is not None:
                cumulative_frame_counts = torch.cumsum(video_grid_thw[:, 0], dim=0)
                cumulative_token_video_counts = torch.cumsum(video_nums, dim=0)
                video_boundary_indices = torch.searchsorted(cumulative_frame_counts, cumulative_token_video_counts)
                video_nums = torch.diff(torch.cat([-video_boundary_indices.new_ones(1), video_boundary_indices]))

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values" and dict_to_expand[key] is not None and image_grid_thw is not None:
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw" and dict_to_expand[key] is not None:
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos" and dict_to_expand[key] is not None and video_grid_thw is not None:
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw" and dict_to_expand[key] is not None:
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "mm_token_type_ids" and dict_to_expand[key] is not None:
                    # mm_token_type_ids has same batch dimension as input_ids
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
                elif key == "pointer_memory_embeds" and dict_to_expand[key] is not None:
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pointer_positions" and dict_to_expand[key] is not None:
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "memory_feat" and dict_to_expand[key] is not None:
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pointer_timestamps" and dict_to_expand[key] is not None:
                    lengths = list(pointer_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )

            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # Expand visual inputs (including pointer memory)
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = ["Qwen3_5ForConditionalGenerationWithPoint3R"]
