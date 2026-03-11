"""
Qwen3.5-VL Processor with Point3R Memory Support.

This module provides a processor for Qwen3.5-VL that handles pointer tokens
for 3D point memory integration. It extends the base Qwen3VLProcessor
(same tokenizer/image processing) with Point3R pointer token support.

Key differences from Qwen3-VL processor:
- Uses mm_token_type_ids to mark pointer tokens as type 1 (same as image tokens)
- No deepstack handling (Qwen3.5 doesn't have deepstack)
"""

from typing import Optional, Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.video_utils import VideoInput


logger = logging.get_logger(__name__)


class Qwen3_5VideosProcessorKwargs(VideosKwargs, total=False):
    pass


class Qwen3_5ImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen3_5ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Qwen3_5ImagesKwargs
    videos_kwargs: Qwen3_5VideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Qwen3_5Processor(ProcessorMixin):
    """
    Qwen3.5-VL processor wrapping image processor, tokenizer, and video processor.

    Mirrors Qwen3VLProcessor but for the Qwen3.5-VL architecture.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3_5ProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen3_5ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3.5-VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Defaulting to `fps=24`."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    @staticmethod
    def _calculate_timestamps(frames_indices, fps, merge_size):
        timestamps = [idx / fps for idx in frames_indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps


class Qwen3_5ProcessorWithPoint3R(Qwen3_5Processor):
    r"""
    Qwen3.5-VL processor with Point3R Memory support.

    Extends Qwen3_5Processor to handle pointer tokens for 3D point memory integration.

    Key differences from Qwen3VLProcessorWithPoint3R:
    - mm_token_type_ids marks pointer tokens as type 1 (like image tokens, for Qwen3.5 position ID computation)
    """

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, pointer_format="video", add_frame_id=False, **kwargs):
        self.pointer_token = "<|pointer_pad|>"
        self.pointer_format = pointer_format
        self.add_frame_id = add_frame_id
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.pointer_token]})
        self.pointer_token_id = self.tokenizer.convert_tokens_to_ids(self.pointer_token)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        pointer_timestamps=None,
        frames_indices=None,
        pointer_fps=None,
        **kwargs: Unpack[Qwen3_5ProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen3_5ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3.5-VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Defaulting to `fps=24`."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # Handle pointer tokens based on pointer_format
        if pointer_timestamps is not None:
            pointer_num_tok = pointer_timestamps.shape[0]

            if self.pointer_format == "video":
                # Video format: grouped by timestamp (like video frames)
                if pointer_fps is not None:
                    fps = pointer_fps
                elif frames_indices is not None:
                    fps = 24.0
                else:
                    fps = 1.0
                unique_timestamps = pointer_timestamps.unique(sorted=True)

                # Build grouped placeholder: <time><vision_start><pointer>*N<vision_end> per group
                pointer_placeholder = ""
                for frame_idx, ts in enumerate(unique_timestamps, start=1):
                    count = (pointer_timestamps == ts).sum().item()

                    if self.add_frame_id:
                        pointer_placeholder += f"<frame-{frame_idx}>"
                    else:
                        ts_val = ts.item()
                        if frames_indices is not None and int(ts_val) < len(frames_indices):
                            time_seconds = frames_indices[int(ts_val)] / fps
                        else:
                            time_seconds = ts_val / fps
                        pointer_placeholder += f"<{time_seconds:.1f} seconds>"

                    pointer_placeholder += (
                        self.vision_start_token + "<|placeholder|>" * count + self.vision_end_token
                    )

                for i in range(len(text)):
                    wrapped = f"{self.vision_start_token}{self.pointer_token}{self.vision_end_token}"
                    if wrapped in text[i]:
                        text[i] = text[i].replace(wrapped, pointer_placeholder, 1)
                    elif self.pointer_token in text[i]:
                        text[i] = text[i].replace(self.pointer_token, pointer_placeholder, 1)
                    text[i] = text[i].replace("<|placeholder|>", self.pointer_token)
            else:
                # Image format: flat expansion
                for i in range(len(text)):
                    while self.pointer_token in text[i]:
                        text[i] = text[i].replace(
                            self.pointer_token,
                            "<|placeholder|>" * pointer_num_tok,
                            1,
                        )
                    text[i] = text[i].replace("<|placeholder|>", self.pointer_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if self.pointer_format == "video" and pointer_timestamps is not None:
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "pointer"])
        else:
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            # Pointer tokens use type 1 (like image) for Qwen3.5 position ID computation
            mm_token_type_ids[array_ids == self.pointer_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)


__all__ = ["Qwen3_5Processor", "Qwen3_5ProcessorWithPoint3R"]
