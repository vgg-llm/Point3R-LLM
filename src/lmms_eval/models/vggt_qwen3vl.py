"""
VGGT-enhanced Qwen3-VL Model Wrapper for lmms-eval

This module implements a wrapper for Qwen3-VL with VGGT geometry encoding support,
enabling 3D scene understanding through on-the-fly geometry feature extraction.

Key Features:
1. **On-the-fly VGGT Encoding**: Extracts 3D geometry features from raw images
2. **Feature Fusion**: Fuses VGGT geometry features with visual features
3. **Qwen3-VL Backend**: Uses Qwen3-VL as the base vision-language model

Usage:
    lmms-eval \\
        --model vggt_qwen3vl \\
        --model_args pretrained=path/to/trained/model \\
        --tasks <task_name> \\
        --batch_size 1
"""

import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import copy
import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    Qwen3VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

from qwen_vl.model.qwen3_vl.modeling_qwen_vggt import Qwen3VLForConditionalGenerationWithVGGT
from qwen_vl.data.utils import load_and_preprocess_images

try:
    from qwen_vl_utils import extract_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("vggt_qwen3vl")
class VGGTQwen3VL(lmms):
    """
    VGGT-enhanced Qwen3-VL Model for lmms-eval

    This model supports on-the-fly 3D geometry encoding via VGGT encoder,
    with geometry features fused with visual features for 3D scene understanding.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_length: Optional[int] = None,
        add_frame_index: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)

        # Determine model class based on config
        if getattr(config, "use_geometry_encoder", False):
            load_class = Qwen3VLForConditionalGenerationWithVGGT
            eval_logger.info("Using Qwen3VLForConditionalGenerationWithVGGT")
        else:
            load_class = Qwen3VLForConditionalGeneration
            eval_logger.info("Using Qwen3VLForConditionalGeneration (no geometry encoder)")

        if use_flash_attention_2:
            self._model = load_class.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = load_class.from_pretrained(
                pretrained,
                config=config,
                torch_dtype="auto",
                device_map=self.device_map
            ).eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            self._model = self.model.to("cuda").to(torch.bfloat16)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for VGGT Qwen3-VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            messages = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        vr = decord.VideoReader(visual)
                        image_num = len(vr)
                        if image_num < self.max_num_frames:
                            frame_indices = np.arange(image_num)
                        else:
                            frame_indices = np.linspace(0, image_num - 1, self.max_num_frames).astype(int)
                        frames = [vr[i].asnumpy() for i in frame_indices]
                        visual_content = []
                        for frame in frames:
                            image = Image.fromarray(frame).convert("RGB")
                            visual_content.append({"type": "image", "image": image})
                        message.append({"role": "user", "content": visual_content + [{"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        image_content = []
                        image_count = 0
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            if self.add_frame_index:
                                image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})
                            image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                            image_count += 1
                        message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                    else:
                        message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

                messages.append(message)

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Prepare geometry encoder inputs if geometry encoder is enabled
            geometry_encoder_inputs = []
            image_inputs = []
            patch_size = self.processor.image_processor.patch_size
            merge_size = self.processor.image_processor.merge_size

            for message in messages:
                vision_info = extract_vision_info(message)
                cur_geometry_encoder_inputs = []
                for ele in vision_info:
                    if "image" in ele:
                        image = ele["image"]
                        if isinstance(image, Image.Image):
                            pass
                        elif isinstance(image, str) and "base64," in image:
                            _, base64_data = image.split("base64,", 1)
                            data = base64.b64decode(base64_data)
                            with BytesIO(data) as bio:
                                image = copy.deepcopy(Image.open(bio))
                        else:
                            raise NotImplementedError("Unsupported image type")
                    else:
                        raise NotImplementedError("Unsupported vision info type")

                    assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                    image = load_and_preprocess_images([image])[0]
                    cur_geometry_encoder_inputs.append(copy.deepcopy(image))
                    _, height, width = image.shape
                    # Adjust dimensions for merge_size alignment
                    if (width // patch_size) % merge_size > 0:
                        width = width - (width // patch_size) % merge_size * patch_size
                    if (height // patch_size) % merge_size > 0:
                        height = height - (height // patch_size) % merge_size * patch_size
                    image = image[:, :height, :width]
                    image_inputs.append(image)

                geometry_encoder_inputs.append(torch.stack(cur_geometry_encoder_inputs))

            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
                do_rescale=False
            )
            device = "cuda" if self.device_map == "auto" else self.device

            # Add geometry encoder inputs if enabled
            if getattr(self.model.config, "use_geometry_encoder", False):
                inputs["geometry_encoder_inputs"] = [feat.to(device) for feat in geometry_encoder_inputs]

            inputs = inputs.to(device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
