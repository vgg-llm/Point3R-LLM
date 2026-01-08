"""
Demo script showing how to use Qwen2.5-VL with Point3R memory.

This demonstrates:
1. Loading the Point3R-enhanced model and processor
2. Processing pointer memory inputs along with images and text
3. Generating responses using pointer tokens
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from qwen_vl.model.modeling_qwen_point3r import Qwen2_5_VLForConditionalGenerationWithPoint3R
from qwen_vl.model.processing_qwen2_5_vl import Qwen2_5_VLProcessorWithPoint3R
from transformers import AutoProcessor
from qwen_vl.model.point3r.point3r import Point3R
from qwen_vl.model.point3r.extract_memory import extract_pointer_memory
from qwen_vl_utils import process_vision_info
from time import time


def main():
    # stage 0, Model loading runtime measurement
    print("\n" + "="*70)
    print(f"Stage 0 (Model Loading)")
    print("="*70)
    stage0_start = time()

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
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    base_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
    )

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

    print(f"\tPointer token: {processor.pointer_token}")
    print(f"\tPointer token ID: {processor.pointer_token_id}")

    ##############################################################################################

    # Load Point3R model for memory extraction
    print("Loading Point3R model...")
    point3r_model = Point3R.from_pretrained("./cache/point3r_512.pth")
    point3r_model = point3r_model.to("cuda")
    point3r_model.eval()

    stage0_end = time()
    print(f"Stage 0 (Model Loading) runtime: {stage0_end - stage0_start:.2f} seconds")

    ##############################################################################################

    # Example 2: Using the model with pointer memory
    print("\n" + "="*70)
    print(f"Stage 1 (Image Feature Pre-processing)")
    print("="*70)

    # stage 1, Image feature pre-processing runtime measurement
    stage1_start = time()

    # vision_message = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image","image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    #         ],
    #     }
    # ]
    vision_message = [
        {
            "role": "user",
            "content": [
                {"type": "image","image": f"./data/demo_data/demo_photos/demo_0{i}.jpg"} for i in range(5)
            ],
        }
    ]

    print("Extracting image info from images...")
    image_inputs, video_inputs = process_vision_info(vision_message)

    # Process images in batches
    batch_size = 2  # Adjust based on available GPU memory
    image_embeds_list = []
    grid_thw_list = []

    for i in range(0, len(image_inputs), batch_size):
        batch_images = image_inputs[i:i+batch_size]

        # Process batch
        processed_batch = processor.image_processor(images=batch_images, min_pixels=min_pixels, max_pixels=max_pixels)

        with torch.inference_mode():
            # Get model device
            model_device = next(model.visual.parameters()).device

            # Move tensors to same device as model (IMPORTANT: reassign the tensor!)
            pixel_values = processed_batch.pixel_values.type(model.visual.dtype)
            pixel_values = pixel_values.to(model_device)
            grid_thw = processed_batch.image_grid_thw

            # print(f'Batch {i//batch_size + 1}: pixel_values shape = {pixel_values.shape}')
            # print(f'Batch {i//batch_size + 1}: grid_thw = {grid_thw}')

            batch_embeds = model.visual(pixel_values, grid_thw=grid_thw)
            # print(f'Batch {i//batch_size + 1}: image_embeds shape = {batch_embeds.shape}')

            image_embeds_list.append(batch_embeds)
            grid_thw_list.append(grid_thw)

    # Concatenate all batches to build equivalent image_embeds
    image_embeds = torch.cat(image_embeds_list, dim=0)
    grid_thw = torch.cat(grid_thw_list, dim=0)

    # print(f'Final concatenated image_embeds shape = {image_embeds.shape}')
    # print(f'Final concatenated grid_thw = {grid_thw}')

    print("Extracting pointer memory from image...")

    # Get the device where Point3R model is located
    point3r_device = next(point3r_model.parameters()).device
    # print(f"Point3R model is on device: {point3r_device}")
    # print(f"Image embeds are on device: {image_embeds.device}")
    # print(f"Grid thw is on device: {grid_thw.device}")

    # Move image_embeds and grid_thw to the same device as Point3R model
    image_embeds = image_embeds.to(point3r_device)
    grid_thw = grid_thw.to(point3r_device)

    # Extract pointer memory from the same image, passing image_embeds and grid_thw
    pointer_data = extract_pointer_memory(
        image_inputs=image_inputs,
        point3r_model=point3r_model,
        image_embeds=image_embeds,
        grid_thw=grid_thw,
        device=point3r_device,
        no_crop=False,
        size=512,
        verbose=False,
    )

    # Free up GPU memory by unloading Point3R model
    print("Unloading Point3R model to free GPU memory...")
    del point3r_model
    torch.cuda.empty_cache()

    stage1_end = time()
    print(f"Stage 1 (Image Feature Pre-processing) runtime: {stage1_end - stage1_start:.2f} seconds")

    # # Save pointer data to file
    # pointer_data_path = "./data/demo_data/pointer_data.pt"
    # print(f"\nSaving pointer data to {pointer_data_path}...")
    # torch.save(pointer_data, pointer_data_path)
    # print("Pointer data saved successfully!")

    # # ============================================================================
    # # CHECKPOINT: You can stop here after Stage 1 and resume from Stage 2 later
    # # ============================================================================

    # # # Load pointer data from file (uncomment to load instead of computing)
    # # pointer_data_path = "./data/demo_data/pointer_data.pt"
    # # print(f"\nLoading pointer data from {pointer_data_path}...")
    # # pointer_data = torch.load(pointer_data_path)
    # # print("Pointer data loaded successfully!")

    # stage 2, LLM runtime measurement
    print("\n" + "="*70)
    print(f"Stage 2 (LLM Run)")
    print("="*70)
    stage2_start = time()

    messages_with_pointer = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<|vision_start|>{processor.pointer_token}<|vision_end|>"},
                {"type": "text", "text": "Describe this scene."},
            ],
        }
    ]

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

    stage2_end = time()
    print(f"Stage 2 (LLM Runtime) runtime: {stage2_end - stage2_start:.2f} seconds")

    print("\n" + "="*70)
    print("Demo completed!")
    print(f"Total runtime: {stage2_end - stage0_start:.2f} seconds")
    print("="*70)

def load_models(load_point3r=True, device=None):

    # stage 0, Model loading runtime measurement
    print("\n" + "="*70)
    print(f"Stage 0 (Model Loading)")
    print("="*70)
    stage0_start = time()

    # Load model with memory-efficient settings
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGenerationWithPoint3R.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        cache_dir="./cache",
        torch_dtype=torch.bfloat16,  # Use bf16 for memory efficiency
        device_map="auto" if device is None else device,  # Automatically distribute model across available devices
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
    )

    # Load the base processor first
    print("Loading processor...")
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    base_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
    )

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

    print(f"\tPointer token: {processor.pointer_token}")
    print(f"\tPointer token ID: {processor.pointer_token_id}")

    ##############################################################################################

    if load_point3r:
        # Load Point3R model for memory extraction
        print("Loading Point3R model...")
        point3r_model = Point3R.from_pretrained("./cache/point3r_512.pth")
        point3r_model = point3r_model.to("cuda" if device is None else device)
        point3r_model.eval()

        stage0_end = time()
        print(f"Stage 0 (Model Loading) runtime: {stage0_end - stage0_start:.2f} seconds")

        return model, processor, min_pixels, max_pixels, point3r_model
    else:
        return model, processor, min_pixels, max_pixels, None

def preprocess_images(
        model,
        processor,
        min_pixels,
        max_pixels,
        point3r_model,
        input_images_dir = "./data/demo_data/demo_photos/", 
        pointer_data_path = "./data/demo_data/pointer_data.pt"
    ):

    # Example 2: Using the model with pointer memory
    print("\n" + "="*70)
    print(f"Stage 1 (Image Feature Pre-processing)")
    print("="*70)

    # stage 1, Image feature pre-processing runtime measurement
    stage1_start = time()

    # Compute sorted list of JPG image paths
    image_paths = sorted(Path(input_images_dir).glob("*.jpg"))
    # Uniformly sample 32 paths
    sample_ct = 32
    if len(image_paths) > sample_ct:
        step = len(image_paths) / sample_ct
        image_paths = [image_paths[int(i * step)] for i in range(sample_ct)]

    vision_message = [
        {
            "role": "user",
            "content": [
                {"type": "image","image": str(img_path)} for img_path in image_paths
            ],
        }
    ]

    print("Extracting image info from images...")
    image_inputs, video_inputs = process_vision_info(vision_message)

    # Process images in batches
    batch_size = 2  # Adjust based on available GPU memory
    image_embeds_list = []
    grid_thw_list = []

    for i in range(0, len(image_inputs), batch_size):
        batch_images = image_inputs[i:i+batch_size]

        # Process batch
        processed_batch = processor.image_processor(images=batch_images, min_pixels=min_pixels, max_pixels=max_pixels)

        with torch.inference_mode():
            # Get model device
            model_device = next(model.visual.parameters()).device

            # Move tensors to same device as model (IMPORTANT: reassign the tensor!)
            pixel_values = processed_batch.pixel_values.type(model.visual.dtype)
            pixel_values = pixel_values.to(model_device)
            grid_thw = processed_batch.image_grid_thw

            # print(f'Batch {i//batch_size + 1}: pixel_values shape = {pixel_values.shape}')
            # print(f'Batch {i//batch_size + 1}: grid_thw = {grid_thw}')

            batch_embeds = model.visual(pixel_values, grid_thw=grid_thw)
            # print(f'Batch {i//batch_size + 1}: image_embeds shape = {batch_embeds.shape}')

            image_embeds_list.append(batch_embeds)
            grid_thw_list.append(grid_thw)

    # Concatenate all batches to build equivalent image_embeds
    image_embeds = torch.cat(image_embeds_list, dim=0)
    grid_thw = torch.cat(grid_thw_list, dim=0)

    # print(f'Final concatenated image_embeds shape = {image_embeds.shape}')
    # print(f'Final concatenated grid_thw = {grid_thw}')

    print("Extracting pointer memory from image...")

    # Get the device where Point3R model is located
    point3r_device = next(point3r_model.parameters()).device
    # print(f"Point3R model is on device: {point3r_device}")
    # print(f"Image embeds are on device: {image_embeds.device}")
    # print(f"Grid thw is on device: {grid_thw.device}")

    # Move image_embeds and grid_thw to the same device as Point3R model
    image_embeds = image_embeds.to(point3r_device)
    grid_thw = grid_thw.to(point3r_device)

    # Extract pointer memory from the same image, passing image_embeds and grid_thw
    pointer_data = extract_pointer_memory(
        image_inputs=image_inputs,
        point3r_model=point3r_model,
        image_embeds=image_embeds,
        grid_thw=grid_thw,
        device=point3r_device,
        no_crop=False,
        size=512,
        verbose=False,
    )

    # Free up GPU memory by unloading Point3R model
    print("Unloading Point3R model to free GPU memory...")
    del point3r_model
    torch.cuda.empty_cache()

    stage1_end = time()
    print(f"Stage 1 (Image Feature Pre-processing) runtime: {stage1_end - stage1_start:.2f} seconds")

    # Save pointer data to file
    print(f"\nSaving pointer data to {pointer_data_path}...")
    torch.save(pointer_data, pointer_data_path)
    print("Pointer data saved successfully!")

def run_models(model,
        processor,
        pointer_data_path = "./data/demo_data/pointer_data.pt"
    ):

    # Load pointer data from file (uncomment to load instead of computing)
    print(f"\nLoading pointer data from {pointer_data_path}...")
    pointer_data = torch.load(pointer_data_path, weights_only=True)
    print("Pointer data loaded successfully!")

    # stage 2, LLM runtime measurement
    print("\n" + "="*70)
    print(f"Stage 2 (LLM Run)")
    print("="*70)
    stage2_start = time()

    messages_with_pointer = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<|vision_start|>{processor.pointer_token}<|vision_end|>"},
                {"type": "text", "text": "Describe this scene."},
            ],
        }
    ]

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

    stage2_end = time()
    print(f"Stage 2 (LLM Runtime) runtime: {stage2_end - stage2_start:.2f} seconds")

    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70)

if __name__=='__main__':
    input_images_dir = "./data/demo_data/demo_photos/"
    pointer_data_path = "./data/demo_data/pointer_data.pt"
    model, processor, min_pixels, max_pixels, point3r_model= load_models()
    preprocess_images(model, processor, min_pixels, max_pixels, point3r_model, input_images_dir, pointer_data_path)
    run_models(model, processor, pointer_data_path)