"""Qwen3-VL with VGGT geometry encoding support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, List, Tuple, Union

from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLCausalLMOutputWithPast
from .configuration_qwen3_vl import Qwen3VLConfig
from ..geometry_encoders import create_geometry_encoder, GeometryEncoderConfig
from ..feature_fusion import FeatureFusionModule, FeatureFusionConfig, GeometryFeatureMerger
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.cache_utils import Cache


class Qwen3VLForConditionalGenerationWithVGGT(Qwen3VLForConditionalGeneration):
    """
    Qwen3-VL model with VGGT 3D geometry encoding support.

    This class extends Qwen3VLForConditionalGeneration to support 3D geometry features
    from VGGT encoder, enabling spatial understanding capabilities.

    Architecture:
    - Inherits from Qwen3VLForConditionalGeneration
    - Adds geometry_encoder (VGGT), geometry_merger, feature_fusion modules
    - Geometry features are fused with visual features AFTER visual embedding extraction
      but BEFORE scatter into inputs_embeds
    """

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)

        # Initialize geometry encoder if enabled
        if getattr(config, 'use_geometry_encoder', False):
            self._init_geometry_encoder(config)

        self.post_init()

    def _init_geometry_encoder(self, config: Qwen3VLConfig):
        """Initialize geometry encoder and related modules."""
        # Create geometry encoder configuration
        encoder_config = GeometryEncoderConfig(
            encoder_type=getattr(config, "geometry_encoder_type", "vggt"),
            model_path=getattr(config, "geometry_encoder_path", None),
            reference_frame=getattr(config, "reference_frame", "first"),
            freeze_encoder=getattr(config, "geometry_encoder_freeze", True)
        )
        self.geometry_encoder = create_geometry_encoder(encoder_config)

        # Create feature merger
        # Note: Qwen3-VL uses config.text_config.hidden_size for LLM hidden dim
        self.geometry_merger = GeometryFeatureMerger(
            output_dim=config.text_config.hidden_size,  # 3584 for Qwen3-VL
            hidden_dim=getattr(config, "geometry_merger_hidden_dim", 4096),
            context_dim=self.geometry_encoder.get_feature_dim(),  # 2048 for VGGT
            spatial_merge_size=config.vision_config.spatial_merge_size,  # 2
            merger_type=getattr(config, "geometry_merger_type", "mlp")
        )

        # Create feature fusion module
        fusion_config = FeatureFusionConfig(
            fusion_method=getattr(config, "feature_fusion_method", "add"),
            hidden_size=config.text_config.hidden_size,
            num_heads=getattr(config, "fusion_attention_heads", 8),
            dropout=getattr(config, "fusion_dropout", 0.1),
            num_layers=getattr(config, "fusion_num_layers", 1)
        )
        self.feature_fusion = FeatureFusionModule(fusion_config)

    def _process_geometry_features(
        self,
        image_embeds: torch.Tensor,
        geometry_encoder_inputs: List[torch.Tensor],
        image_grid_thw: torch.LongTensor
    ) -> torch.Tensor:
        """Process geometry features using the geometry encoder.

        Args:
            image_embeds: Visual embeddings from Qwen3 vision tower
                         Shape: (total_tokens, hidden_size) where total_tokens = sum of all image tokens
            geometry_encoder_inputs: List[Tensor] - raw images for each batch item
                                    Each tensor: (n_images, C, H, W)
            image_grid_thw: Grid dimensions (T, H_pre_merge, W_pre_merge) for each image from Qwen3-VL
                           Shape: (num_images, 3) - used to calculate target spatial dims

        Returns:
            Fused features: (total_tokens, hidden_size)
        """
        # Validate: total geometry images must match image_grid_thw rows
        total_geo_images = sum(inp.shape[0] for inp in geometry_encoder_inputs)
        assert total_geo_images == image_grid_thw.shape[0], (
            f"Geometry encoder inputs have {total_geo_images} total images, "
            f"but image_grid_thw has {image_grid_thw.shape[0]} entries. "
            f"Every batch item with images must provide matching geometry inputs."
        )

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        batch_size = len(geometry_encoder_inputs)
        geo_embeds_interpolated = []
        img_idx = 0

        for bn in range(batch_size):
            n_image = geometry_encoder_inputs[bn].shape[0]
            if n_image == 0:
                continue

            _, _, height, width = geometry_encoder_inputs[bn].shape

            # Encode geometry features using VGGT (patch_size=14)
            # Output shape depends on encoder, reshaped to (n_image, h_patch, w_patch, feat_dim)
            features = self.geometry_encoder.encode(geometry_encoder_inputs[bn])
            features = features.to(image_embeds.dtype)

            h_patch = height // self.geometry_encoder.patch_size
            w_patch = width // self.geometry_encoder.patch_size
            features = features.reshape(n_image, h_patch, w_patch, -1)

            # Apply merger: (n_image, h_patch//2, w_patch//2, hidden_size)
            features = self.geometry_merger(features)

            # Interpolate each image to match Qwen3-VL's post-merge spatial dims
            for i in range(n_image):
                feat = features[i]  # (h_geo, w_geo, hidden)
                t, h, w = image_grid_thw[img_idx].tolist()
                target_h = h // spatial_merge_size
                target_w = w // spatial_merge_size

                # Reshape for interpolation: (1, hidden, h_geo, w_geo)
                feat_for_interp = feat.permute(2, 0, 1).unsqueeze(0)

                feat_interp = F.interpolate(
                    feat_for_interp.float(),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).to(feat.dtype)

                # (target_h * target_w, hidden)
                feat_flat = feat_interp.squeeze(0).permute(1, 2, 0).reshape(-1, feat.shape[-1])

                # Replicate for temporal frames
                if t > 1:
                    feat_flat = feat_flat.repeat(t, 1)  # (t * target_h * target_w, hidden)

                geo_embeds_interpolated.append(feat_flat)
                img_idx += 1

        if geo_embeds_interpolated:
            geo_embeds = torch.cat(geo_embeds_interpolated, dim=0)
            assert geo_embeds.shape[0] == image_embeds.shape[0], (
                f"Geometry token count ({geo_embeds.shape[0]}) != "
                f"image token count ({image_embeds.shape[0]})"
            )
            image_embeds = self.feature_fusion(image_embeds, geo_embeds)

        return image_embeds

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model with optional geometry encoder checkpoint."""
        geometry_encoder_path = kwargs.pop("geometry_encoder_path", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if geometry_encoder_path and hasattr(model, 'geometry_encoder'):
            model.geometry_encoder.load_model(geometry_encoder_path)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        geometry_encoder_inputs: Optional[List[torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, Qwen3VLCausalLMOutputWithPast]:
        """
        Forward pass with geometry encoder support.

        Args:
            geometry_encoder_inputs: List of raw image tensors for geometry encoding.
                                    Each tensor: (n_images, C, H, W)
            ... (other args same as parent)
        """
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Handle image embeddings with geometry fusion
        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            # Get visual features (with DeepStack)
            image_embeds, deepstack_image_embeds = self.model.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            # GEOMETRY FUSION: Process 3D geometry features if enabled
            if (getattr(self.config, 'use_geometry_encoder', False)
                and geometry_encoder_inputs is not None):
                image_embeds = self._process_geometry_features(
                    image_embeds, geometry_encoder_inputs, image_grid_thw
                )

            # Get mask and scatter
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            # Get video features (with DeepStack)
            video_embeds, deepstack_video_embeds = self.model.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            # Get mask and scatter
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Aggregate visual position masks and deepstack embeddings
        visual_pos_masks = None
        deepstack_visual_embeds = None

        if image_mask is not None and video_mask is not None:
            # Aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # Calculate position IDs if not provided
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Forward through language model
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Compute logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

        return Qwen3VLCausalLMOutputWithPast(
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
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        geometry_encoder_inputs=None,
        **kwargs,
    ):
        """Prepare inputs for generation with geometry encoder support."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Add geometry encoder inputs
        model_inputs["geometry_encoder_inputs"] = geometry_encoder_inputs

        # Position IDs handled in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            # After the prefill phase, visual inputs should not be forwarded
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["geometry_encoder_inputs"] = None

        return model_inputs

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expand inputs for beam search with geometry encoder support."""
        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values", "image_grid_thw", "pixel_values_videos",
            "video_grid_thw", "second_per_grid_ts", "geometry_encoder_inputs",
        ]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

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
                elif key == "second_per_grid_ts" and dict_to_expand[key] is not None:
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
                elif key == "geometry_encoder_inputs" and dict_to_expand[key] is not None:
                    # geometry_encoder_inputs is List[Tensor], one per batch item
                    expanded = []
                    for item in dict_to_expand[key]:
                        for _ in range(expand_size):
                            expanded.append(item)
                    dict_to_expand[key] = expanded
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


__all__ = ["Qwen3VLForConditionalGenerationWithVGGT"]
