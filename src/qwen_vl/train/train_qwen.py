# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig, set_seed, enable_full_determinism

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    if model_args.use_geometry_encoder:
        # vggt is frozen
        for n, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False
    
    if model_args.use_pointer_memory and not model_args.use_preprocessed_input:
        # point3r memory is frozen
        for n, p in model.point3r_model.named_parameters():
            p.requires_grad = False

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    print('='*70)
    if "qwen2.5" in model_args.model_name_or_path.lower():
        if getattr(model_args, "use_pointer_memory", False):
            from qwen_vl.model.modeling_qwen_point3r import Qwen2_5_VLForConditionalGenerationWithPoint3R
            from qwen_vl.model.processing_qwen2_5_vl import Qwen2_5_VLProcessorWithPoint3R
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            if hasattr(config, "use_pointer_memory") and config.use_pointer_memory != model_args.use_pointer_memory:
                raise ValueError(
                    "The use_pointer_memory in config and model_args are not consistent. "
                    "Please check the model config."
                )
            # TODO: do we need all these attributes?
            for k in [
                "use_pointer_memory",
                "use_preprocessed_input",
                "point3r_model_path",
                "pointer_memory_size"
            ]:
                setattr(config, k, getattr(model_args, k))

            assert model_args.use_preprocessed_input or model_args.point3r_model_path is not None, \
                "When use_pointer_memory is True, use_preprocessed_input must be True or point3r_model_path must be set in the config."
            model = Qwen2_5_VLForConditionalGenerationWithPoint3R.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                point3r_model_path=model_args.point3r_model_path
            )

            base_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                use_fast=True,
                cache_dir=training_args.cache_dir,
                min_pixels=data_args.min_pixels, 
                max_pixels=data_args.max_pixels
            )

            # Create Point3R processor with pointer token support
            processor = Qwen2_5_VLProcessorWithPoint3R(
                image_processor=base_processor.image_processor,
                tokenizer=base_processor.tokenizer,
                chat_template=base_processor.chat_template if hasattr(base_processor, 'chat_template') else None,
            )

            # Store pointer token ID in model config for proper processing
            model.config.pointer_token_id = processor.pointer_token_id
            model.pointer_token_id = processor.pointer_token_id

            # Resize token embeddings to accommodate new pointer token
            model.resize_token_embeddings(len(processor.tokenizer))

            data_args.image_processor = processor.image_processor
            data_args.model_type = "qwen2.5vl"

        elif model_args.use_geometry_encoder:
            from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            for k in [
                "use_geometry_encoder", 
                "geometry_encoder_type", 
                "reference_frame",
                "feature_fusion_method", 
                "fusion_num_layers",
                "geometry_merger_type"
            ]:
                setattr(config, k, getattr(model_args, k))
            print(f'config: {config}')

            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                geometry_encoder_path=model_args.geometry_encoder_path
            )
            data_args.image_processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path
                ).image_processor
            data_args.model_type = "qwen2.5vl"
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            data_args.image_processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path
                ).image_processor
            data_args.model_type = "qwen2.5vl"

    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.use_pointer_memory:
        tokenizer = processor.tokenizer
        assert tokenizer.padding_side == "right", "Padding side must be right"
        # Q. does use_fast make any difference?
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)

    # Pass processor if using Point3R, otherwise pass None
    processor_to_pass = processor if model_args.use_pointer_memory else None
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, processor=processor_to_pass)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    # Try to copy chat_template.json if it exists
    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    if os.path.exists(source_path):
        shutil.copy2(source_path, template_path)
    else:
        # Try to find it in the cache directory
        cache_pattern = os.path.join(training_args.cache_dir or "./cache", f"models--{model_args.model_name_or_path.replace('/', '--')}", "**", "chat_template.json")
        import glob
        cached_files = glob.glob(cache_pattern, recursive=True)
        if cached_files:
            shutil.copy2(cached_files[0], template_path)
        else:
            rank0_print(f"Warning: chat_template.json not found at {source_path} or in cache, skipping copy")

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
