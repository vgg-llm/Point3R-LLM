"""
Demo script showing how to use Qwen2.5-VL with Point3R memory.

This demonstrates:
1. Loading the Point3R-enhanced model and processor
2. Processing pointer memory inputs along with images and text
3. Generating responses using pointer tokens
"""

import torch
import sys
sys.path.insert(0, 'src')

from qwen_vl.model.modeling_qwen_point3r import Qwen2_5_VLForConditionalGenerationWithPoint3R
from qwen_vl.model.processing_qwen2_5_vl import Qwen2_5_VLProcessorWithPoint3R
from transformers import AutoProcessor
from qwen_vl.model.point3r.point3r import Point3R
from qwen_vl_utils import process_vision_info
# from omegaconf import OmegaConf
# point3r_config = OmegaConf.load("./config/point3r_finetune.yaml")
# OmegaConf.resolve(point3r_config)  # Resolve interpolations
# self.pointer_memory = Point3R(point3r_config)

# Load model with memory-efficient settings
print("Loading model...")
model = Qwen2_5_VLForConditionalGenerationWithPoint3R.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir="./cache",
    torch_dtype=torch.bfloat16,  # Use bf16 for memory efficiency
    device_map="auto",  # Automatically distribute model across available devices
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
)

# Load the base processor first
print("Loading processor...")
base_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)

# Create Point3R processor with pointer token support
processor = Qwen2_5_VLProcessorWithPoint3R(
    image_processor=base_processor.image_processor,
    tokenizer=base_processor.tokenizer,
    chat_template=base_processor.chat_template if hasattr(base_processor, 'chat_template') else None,
)

##################### This part should be inside Qwen2_5_VLForConditionalGenerationWithPoint3R 

# Store pointer token ID in model config for proper processing
model.config.pointer_token_id = processor.pointer_token_id
model.pointer_token_id = processor.pointer_token_id

# Resize token embeddings to accommodate new pointer token
model.resize_token_embeddings(len(processor.tokenizer))

print(f"Pointer token: {processor.pointer_token}")
print(f"Pointer token ID: {processor.pointer_token_id}")

##############################################################################################

# # Example 1: Using the model with standard image input (no pointers)
# print("\n" + "="*70)
# print("Example 1: Standard Image Input (without pointers)")
# print("="*70)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "What is the type of the Dog?"},
#         ],
#     }
# ]

# # Process inputs without pointers
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# image_inputs, video_inputs = process_vision_info(messages)

# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

# # Move inputs to the same device as the model
# inputs = inputs.to(model.device)

# # Generate with memory-efficient settings
# print("Generating response...")
# with torch.inference_mode():
#     generated_ids = model.generate(
#         **inputs,
#         max_new_tokens=128,
#         do_sample=False,  # Greedy decoding to save memory
#     )

# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(f"Response: {output_text[0]}")


##############################################################################################

# Example 2: Using the model with pointer memory
print("\n" + "="*70)
print("Example 2: With Pointer Memory")
print("="*70)

# Load Point3R model for memory extraction
print("Loading Point3R model...")
point3r_model = Point3R.from_pretrained("./cache/point3r_512.pth")
point3r_model = point3r_model.to("cuda")
point3r_model.eval()


messages_with_pointer = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"<|vision_start|>{processor.pointer_token}<|vision_end|>"},
            {"type": "text", "text": "Describe this scene."},
        ],
    }
]

vision_info = [
    {
        "role": "user",
        "content": [
            {"type": "image","image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        ],
    }
]
# vision_info = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image","image": f"./data/demo_data/demo_photos/demo_0{i}.jpg"} for i in range(1)
#         ],
#     }
# ]

print("Extracting image info from image...")
image_inputs, video_inputs = process_vision_info(vision_info)
image_text = processor.apply_chat_template(vision_info, tokenize=False, add_generation_prompt=True)
inputs_for_images = processor(
    text=[image_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
with torch.inference_mode():
    pixel_values = inputs_for_images.pixel_values.type(model.visual.dtype)
    grid_thw = inputs_for_images.image_grid_thw
    print(f'grid_thw = {grid_thw}')
    image_embeds = model.visual(pixel_values, grid_thw=grid_thw)

print("Extracting pointer memory from image...")
from extract_pointer_memory import extract_pointer_memory

# Extract pointer memory from the same image, passing image_embeds and grid_thw
pointer_data = extract_pointer_memory(
    image_inputs=image_inputs,
    point3r_model=point3r_model,
    image_embeds=image_embeds,
    grid_thw=grid_thw,
    device="cuda",
    no_crop=False,
    size=512,
    verbose=True,
)

# Create message with pointer token
print("\nGenerating response with pointer memory...")
text_pointer = processor.apply_chat_template(messages_with_pointer, tokenize=False, add_generation_prompt=True)
inputs_pointer = processor(
    text=[text_pointer],
    pointers=pointer_data['pointer_memory_embeds'],
    padding=True,
    return_tensors="pt",
)

# Move inputs to device
inputs_pointer = inputs_pointer.to(model.device)
pointer_memory_embeds = pointer_data['pointer_memory_embeds'].to(model.device)
pointer_positions = pointer_data['pointer_positions'].to(model.device)

# Use properly computed embeddings from extract_pointer_memory
pointer_aligned_image_embeds = pointer_data['pointer_memory_embeds']

# Verify shapes match
assert pointer_aligned_image_embeds.shape[0] == pointer_positions.shape[0], \
    f"Shape mismatch: embeds {pointer_aligned_image_embeds.shape} vs positions {pointer_positions.shape}"

# Generate with pointer memory
with torch.inference_mode():
    generated_ids_pointer = model.generate(
        **inputs_pointer,
        pointer_memory_embeds=pointer_aligned_image_embeds,
        pointer_positions=pointer_positions,
        max_new_tokens=128,
        do_sample=True,
    )

generated_ids_pointer_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_pointer.input_ids, generated_ids_pointer)
]
output_text_pointer = processor.batch_decode(
    generated_ids_pointer_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Response with pointer memory: {output_text_pointer[0]}")

print("\n" + "="*70)
print("Demo completed!")
print("="*70)
