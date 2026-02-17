"""
Demo script showing how to use Qwen2.5-VL with Point3R memory.

This demonstrates:
1. Loading the Point3R-enhanced model and processor
2. Processing pointer memory inputs along with images and text
3. Generating responses using pointer tokens
"""

import torch
import sys
import os
from pathlib import Path
sys.path.insert(0, 'src')
from natsort import natsorted

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

    model_path = "Qwen/Qwen3-VL-4B-Instruct"
    # Load model with memory-efficient settings
    print("Loading model...")
    pointer_format = "video"  # Default pointer format for main() demo
    if 'Qwen3-VL' in model_path:
        from qwen_vl.model.qwen3_vl.modeling_qwen3_point3r import Qwen3VLForConditionalGenerationWithPoint3R
        from qwen_vl.model.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorWithPoint3R
        # Load model with memory-efficient settings
        print(f"Loading model from: {model_path}")
        model = Qwen3VLForConditionalGenerationWithPoint3R.from_pretrained(
            model_path,
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
            model_path, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
        )

        # Create Point3R processor with pointer token support
        processor = Qwen3VLProcessorWithPoint3R(
            image_processor=base_processor.image_processor,
            tokenizer=base_processor.tokenizer,
            video_processor=base_processor.video_processor,
            chat_template=base_processor.chat_template if hasattr(base_processor, 'chat_template') else None,
            pointer_format=pointer_format,
        )
    else:
        from qwen_vl.model.modeling_qwen_point3r import Qwen2_5_VLForConditionalGenerationWithPoint3R
        from qwen_vl.model.processing_qwen2_5_vl import Qwen2_5_VLProcessorWithPoint3R

        # Load model with memory-efficient settings
        print(f"Loading model from: {model_path}")
        model = Qwen2_5_VLForConditionalGenerationWithPoint3R.from_pretrained(
            model_path,
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
            model_path, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
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
        no_crop=True,
        size=512,
        verbose=True,
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
    memory_aligned_image_embeds = pointer_data['pointer_memory_embeds']

    # Verify shapes match
    assert memory_aligned_image_embeds.shape[0] == pointer_positions.shape[0], \
        f"Shape mismatch: embeds {memory_aligned_image_embeds.shape} vs positions {pointer_positions.shape}"

    # Generate with pointer memory
    with torch.inference_mode():
        generated_ids_pointer = model.generate(
            **inputs_pointer,
            pointer_memory_embeds=memory_aligned_image_embeds,
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

def load_models(load_point3r=True, device=None, model_path="Qwen/Qwen2.5-VL-3B-Instruct", pointer_format="video"):
    """
    Load models for inference.

    Args:
        load_point3r: Whether to load Point3R model for memory extraction
        device: Device to load models on (default: auto)
        model_path: Path to model checkpoint or HuggingFace model ID
                   Examples:
                   - "Qwen/Qwen2.5-VL-3B-Instruct" (base model)
                   - "outputs/scan2cap_point3r_all_frames" (fine-tuned)

    Returns:
        model, processor, min_pixels, max_pixels, point3r_model (or None)
    """
    # stage 0, Model loading runtime measurement
    print("\n" + "="*70)
    print(f"Stage 0 (Model Loading)")
    print("="*70)
    stage0_start = time()

    if 'Qwen3-VL' in model_path or 'Qwen3VL' in model_path:
        from qwen_vl.model.qwen3_vl.modeling_qwen3_point3r import Qwen3VLForConditionalGenerationWithPoint3R
        from qwen_vl.model.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorWithPoint3R
        # Load model with memory-efficient settings
        print(f"Loading model from: {model_path}")
        model = Qwen3VLForConditionalGenerationWithPoint3R.from_pretrained(
            model_path,
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
            model_path, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
        )

        # Create Point3R processor with pointer token support
        processor = Qwen3VLProcessorWithPoint3R(
            image_processor=base_processor.image_processor,
            tokenizer=base_processor.tokenizer,
            video_processor=base_processor.video_processor,
            chat_template=base_processor.chat_template if hasattr(base_processor, 'chat_template') else None,
            pointer_format=pointer_format,
        )
    else:
        from qwen_vl.model.modeling_qwen_point3r import Qwen2_5_VLForConditionalGenerationWithPoint3R
        from qwen_vl.model.processing_qwen2_5_vl import Qwen2_5_VLProcessorWithPoint3R

        # Load model with memory-efficient settings
        print(f"Loading model from: {model_path}")
        model = Qwen2_5_VLForConditionalGenerationWithPoint3R.from_pretrained(
            model_path,
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
            model_path, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
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
        pointer_data_path = None,
        use_viser = False,
        unload_point3r_model = False,
        annotation_result = None,
        input_poses_dir = None,
        scannet_pth_path = None,
        lambda_decay = 1.0,
        sample_ct = 32,
        max_memory_tokens = None,
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG"),
    ):

    # Example 2: Using the model with pointer memory
    print("\n" + "="*70)
    print(f"Stage 1 (Image Feature Pre-processing)")
    print("="*70)

    # stage 1, Image feature pre-processing runtime measurement
    stage1_start = time()

    # Compute sorted list of image paths
    p = Path(input_images_dir)
    image_paths = natsorted([f for ext in image_extensions for f in p.glob(ext)])
    # Uniformly sample 32 paths
    
    if len(image_paths) > sample_ct:
        step = len(image_paths) / sample_ct
        frames_indices = [int(i * step) for i in range(sample_ct)]
        image_paths = [image_paths[idx] for idx in frames_indices]
    else:
        frames_indices = list(range(len(image_paths)))

    # Read pose files from input_poses_dir if provided
    pose_paths = None
    if input_poses_dir is not None:
        poses_p = Path(input_poses_dir)
        pose_paths = natsorted(list(poses_p.glob("*.txt")))
        if len(pose_paths) > 0:
            print(f"Found {len(pose_paths)} pose files in {input_poses_dir}")
        else:
            print(f"Warning: No .txt pose files found in {input_poses_dir}")
            pose_paths = None
        if len(pose_paths) > sample_ct:
            step = len(pose_paths) / sample_ct
            pose_paths = [pose_paths[int(i * step)] for i in range(sample_ct)]

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
    deepstack_image_embeds = []  # List of lists: [[layer0_batch0, layer0_batch1, ...], [layer1_batch0, ...], ...]
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

            visual_output = model.visual(pixel_values, grid_thw=grid_thw)
            batch_deepstack_image_embeds = None
            if isinstance(visual_output, tuple):
                batch_image_embeds, batch_deepstack_image_embeds = visual_output
            else:
                batch_image_embeds = visual_output

            # print(f'Batch {i//batch_size + 1}: image_embeds shape = {batch_embeds.shape}')

            image_embeds_list.append(batch_image_embeds)
            if batch_deepstack_image_embeds is not None:
                # Initialize list structure on first batch
                if len(deepstack_image_embeds) == 0:
                    deepstack_image_embeds = [[] for _ in range(len(batch_deepstack_image_embeds))]
                # Append each layer's embeddings to corresponding list
                for layer_idx, layer_embeds in enumerate(batch_deepstack_image_embeds):
                    deepstack_image_embeds[layer_idx].append(layer_embeds)
            grid_thw_list.append(grid_thw)

    # Concatenate all batches to build equivalent image_embeds
    image_embeds = torch.cat(image_embeds_list, dim=0)
    grid_thw = torch.cat(grid_thw_list, dim=0)

    # Concatenate deepstack embeddings for each layer
    if deepstack_image_embeds:
        deepstack_image_embeds = [
            torch.cat(layer_embeds_list, dim=0) for layer_embeds_list in deepstack_image_embeds
        ]

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

    # Move deepstack embeddings to Point3R device if available
    if deepstack_image_embeds:
        deepstack_image_embeds = [layer.to(point3r_device) for layer in deepstack_image_embeds]

    # Extract pointer memory from the same image, passing image_embeds and grid_thw
    pointer_data = extract_pointer_memory(
        image_inputs=image_inputs,
        point3r_model=point3r_model,
        image_embeds=image_embeds,
        grid_thw=grid_thw,
        deepstack_image_embeds=deepstack_image_embeds if deepstack_image_embeds else None,
        device=point3r_device,
        no_crop=False,
        size=512,
        verbose=True,
        use_viser=use_viser,
        annotation_result=annotation_result,
        scannet_pth_path=scannet_pth_path,
        scannet_pose_paths=pose_paths,
        lambda_decay=lambda_decay,
        max_memory_tokens=max_memory_tokens,
        frames_indices=frames_indices,
    )

    # Log timestamp statistics for testing
    if 'pointer_timestamps' in pointer_data:
        timestamps = pointer_data['pointer_timestamps']
        print(f"\n[Timestamp Tracking]")
        print(f"  - Total tokens: {timestamps.shape[0]}")
        print(f"  - Timestamp range: [{timestamps.min().item()}, {timestamps.max().item()}]")
        print(f"  - Unique timestamps: {timestamps.unique().tolist()}")
        # Show distribution per timestamp
        print(f"tokens count: ", end="")
        for ts in timestamps.unique().tolist():
            count = (timestamps == ts).sum().item()
            # print(f"  - Frame {ts}: {count} tokens")
            print(count, end=", ")
        print()

    if unload_point3r_model:
        # Free up GPU memory by unloading Point3R model
        print("Unloading Point3R model to free GPU memory...")
        del point3r_model
    torch.cuda.empty_cache()

    stage1_end = time()
    print(f"Stage 1 (Image Feature Pre-processing) runtime: {stage1_end - stage1_start:.2f} seconds")

    # Save pointer data to file if path is provided
    if pointer_data_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(pointer_data_path), exist_ok=True)
        print(f"\nSaving pointer data to {pointer_data_path}...")
        torch.save(pointer_data, pointer_data_path)
        print("Pointer data saved successfully!")
    else:
        print("\nSkipping save (pointer_data_path is None)")

    # Return pointer_data for in-memory use
    return pointer_data

def run_models(model,
        processor,
        pointer_data_path = "./data/demo_data/pointer_data.pt",
        query = "Describe this scene.",
        pointer_data = None
    ):

    # Load pointer data from file if not provided
    if pointer_data is None:
        print(f"\nLoading pointer data from {pointer_data_path}...")
        pointer_data = torch.load(pointer_data_path, weights_only=True)
        print("Pointer data loaded successfully!")
    else:
        print(f"\nUsing pre-loaded pointer data")

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
                {"type": "text", "text": query},
            ],
        }
    ]

    # Create message with pointer token
    print("\nGenerating response with pointer memory...")
    text_pointer = processor.apply_chat_template(messages_with_pointer, tokenize=False, add_generation_prompt=True)
    inputs_pointer = processor(
        text=[text_pointer],
        pointer_timestamps=pointer_data['pointer_timestamps'],
        frames_indices=pointer_data.get('frames_indices'),
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to device
    inputs_pointer = inputs_pointer.to(model.device)
    pointer_memory_embeds = pointer_data['pointer_memory_embeds'].to(model.device)
    pointer_positions = pointer_data['pointer_positions'].to(model.device)

    # Load deepstack embeddings if available
    deepstack_pointer_embeds = None
    if 'deepstack_image_embeds' in pointer_data:
        deepstack_pointer_embeds = [layer.to(model.device) for layer in pointer_data['deepstack_image_embeds']]

    # Verify shapes match
    assert pointer_memory_embeds.shape[0] == pointer_positions.shape[0], \
        f"Shape mismatch: embeds {pointer_memory_embeds.shape} vs positions {pointer_positions.shape}"

    # Prepare pointer timestamps if available
    pointer_timestamps = None
    if 'pointer_timestamps' in pointer_data:
        pointer_timestamps = pointer_data['pointer_timestamps'].to(model.device)
        
    # Debug hook to capture rope index
    _original_get_rope_index = model.get_rope_index

    def _debug_get_rope_index(*args, **kwargs):
        position_ids, rope_deltas = _original_get_rope_index(*args, **kwargs)
        print(f"\n[DEBUG RoPE] position_ids shape: {position_ids.shape}")
        print(f"[DEBUG RoPE] rope_deltas: {rope_deltas}")
        print(f"[DEBUG RoPE] position_ids range per dim:")
        for d in range(position_ids.shape[0]):
            vals = position_ids[d, 0]
            print(f"  dim {d}: min={vals.min().item()}, max={vals.max().item()}")
        print(position_ids[0, 0][:100])
        print(position_ids[0, 0][4000:4100])
        print(position_ids[0, 0][-100:])
        # Save for inspection
        return position_ids, rope_deltas

    model.get_rope_index = _debug_get_rope_index

    # Generate with pointer memory
    with torch.inference_mode():
        generated_ids_pointer = model.generate(
            **inputs_pointer,
            pointer_memory_embeds=pointer_memory_embeds,
            pointer_positions=pointer_positions,
            deepstack_pointer_embeds=deepstack_pointer_embeds,
            pointer_timestamps=pointer_timestamps,
            max_new_tokens=1024,
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

    # print("\n" + "="*70)
    # print("Demo completed!")
    # print("="*70)

    # Return the generated response
    return output_text_pointer[0]

def extract_scene_id_from_pointer_path(pointer_data_path: str) -> str:
    """
    Extract scene_id from pointer_data path.
    Example: "scannet/pointer_memory/scene0000_00.pt" -> "scene0000_00"

    Args:
        pointer_data_path: Relative path to pointer data file

    Returns:
        Scene ID extracted from the filename
    """
    # Get filename without extension
    filename = os.path.basename(pointer_data_path)
    scene_id = os.path.splitext(filename)[0]
    return scene_id


def extract_box_and_coordinates_from_scan2cap(
    annotation_path: str,
    reference_frame: str = "first",
    verify_transform: bool = True
) -> dict:
    """
    Extract gt_box and transformed box center coordinates from scan2cap annotation file.

    Each element in the annotation file contains:
    - conversations[0]['value']: Contains a coordinate [x, y, z] in the text
    - gt_box: 6-element bounding box [x_min, y_min, z_min, x_max, y_max, z_max]
    - input_box: The box used for transformation (gt_box for train, pred_box for val)
    - cam2img, cam2global, axis_align_matrix: Transformation matrices

    Elements sharing the same pointer_data share the same transformation matrices.

    The transformation from input_box[:3] (global coordinates) to transformed_center
    (camera coordinates) is computed as:
        extrinsic = axis_align_matrix @ reference_frame_cam2global
        global2cam = inv(extrinsic)
        transformed_coord = global2cam @ [input_box[0], input_box[1], input_box[2], 1]

    Args:
        annotation_path: Path to the scan2cap annotation JSON file
        reference_frame: Which frame to use as reference ("first" or "last")
        verify_transform: If True, verify that computed transform matches stored value

    Returns:
        dict with structure:
        {
            'by_pointer_data': {
                '<pointer_data_path>': {
                    'cam2img': [[...]],
                    'cam2global': [[[...]]],
                    'axis_align_matrix': [[...]],
                    'elements': [
                        {
                            'gt_box': [x_min, y_min, z_min, x_max, y_max, z_max],
                            'input_box': [...],
                            'box_center': [x, y, z],  # input_box[:3] used for transform
                            'transformed_center': [x, y, z],  # from conversation text
                            'computed_transformed_center': [x, y, z],  # recomputed
                            'transform_matches': bool,
                            'metadata': {...},
                            'query': '...'
                        },
                        ...
                    ]
                },
                ...
            },
            'all_elements': [...]  # flat list of all elements with their data
        }
    """
    import json
    import re
    import numpy as np

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    result = {
        'by_pointer_data': {},
        'all_elements': []
    }

    # Regex pattern to extract [x, y, z] coordinates from conversation text
    coord_pattern = r'\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]'

    for idx, sample in enumerate(annotations):
        pointer_data = sample.get('pointer_data', '')

        # Initialize pointer_data entry if not exists
        if pointer_data not in result['by_pointer_data']:
            result['by_pointer_data'][pointer_data] = {
                'cam2img': sample.get('cam2img', []),
                'cam2global': sample.get('cam2global', []),
                'axis_align_matrix': sample.get('axis_align_matrix', []),
                'elements': []
            }

        # Extract gt_box and input_box
        gt_box = sample.get('gt_box', [])
        input_box = sample.get('input_box', [])

        # box_center is input_box[:3] - this is what gets transformed
        # (matches original code: input_box[:3] + [1])
        box_center = None
        if len(input_box) >= 3:
            box_center = input_box[:3]

        # Also compute gt_box center for reference
        gt_box_center = None
        if len(gt_box) >= 3:
            gt_box_center = gt_box[:3]

        # Extract transformed center coordinate from conversation text
        transformed_center = None
        conversation_value = sample.get('conversations', [{}])[0].get('value', '')
        coord_match = re.search(coord_pattern, conversation_value)
        if coord_match:
            transformed_center = [
                float(coord_match.group(1)),
                float(coord_match.group(2)),
                float(coord_match.group(3))
            ]

        # Compute the transformation to verify
        computed_transformed_center = None
        cam2global_list = sample.get('cam2global', [])
        axis_align_matrix = sample.get('axis_align_matrix', [])

        ref_cam2global = None
        if box_center and cam2global_list and axis_align_matrix:
            # Select reference frame (first or last)
            ref_cam2global = cam2global_list[0] if reference_frame == "first" else cam2global_list[-1]

            # Compute extrinsic: axis_align_matrix @ cam2global
            axis_align_np = np.array(axis_align_matrix)
            ref_cam2global_np = np.array(ref_cam2global)
            extrinsic = axis_align_np @ ref_cam2global_np

            # Compute global2cam (inverse of extrinsic)
            global2cam = np.linalg.inv(extrinsic)

            # Transform box center from global to camera coordinates
            box_center_homogeneous = np.array(box_center + [1.0]).reshape(4, 1)
            transformed = (global2cam @ box_center_homogeneous).reshape(4)[:3]
            computed_transformed_center = [round(x, 2) for x in transformed.tolist()]

        # Extract query (text after the coordinate)
        query = ''
        if '<|vision_end|>' in conversation_value:
            query = conversation_value.split('<|vision_end|>')[-1].strip()

        # Verify transformation matches if requested
        transform_matches = None
        if verify_transform and transformed_center and computed_transformed_center:
            transform_matches = (transformed_center == computed_transformed_center)

        element_data = {
            'index': idx,
            'gt_box': gt_box,
            'gt_box_center': gt_box_center,
            'input_box': input_box,
            'box_center': box_center,  # input_box[:3] used for transformation
            'transformed_center': transformed_center,  # from conversation text
            'computed_transformed_center': computed_transformed_center,  # recomputed
            'transform_matches': transform_matches,
            'metadata': sample.get('metadata', {}),
            'query': query,
            'iou': sample.get('iou', None),
            'pointer_data': pointer_data,
            'global2cam': global2cam,
            'ref_cam2global': ref_cam2global_np,
            'axis_align': axis_align_np
        }

        result['by_pointer_data'][pointer_data]['elements'].append(element_data)
        result['all_elements'].append(element_data)

    return result

def print_extracted_data_summary(extracted_data: dict):
    """
    Print a summary of extracted box and coordinate data.

    Args:
        extracted_data: Output from extract_box_and_coordinates_from_scan2cap
    """
    print("\n" + "="*70)
    print("Extracted Box and Coordinate Data Summary")
    print("="*70)

    print(f"\nTotal elements: {len(extracted_data['all_elements'])}")
    print(f"Unique scenes (pointer_data): {len(extracted_data['by_pointer_data'])}")

    # Count transform verification results
    all_matches = [e.get('transform_matches') for e in extracted_data['all_elements'] if e.get('transform_matches') is not None]
    if all_matches:
        match_count = sum(all_matches)
        print(f"Transform verification: {match_count}/{len(all_matches)} matched")

    for pointer_data, scene_data in extracted_data['by_pointer_data'].items():
        print(f"\n{'-'*70}")
        print(f"Scene: {pointer_data}")
        print(f"Number of elements: {len(scene_data['elements'])}")
        print(f"cam2img shape: {len(scene_data['cam2img'])}x{len(scene_data['cam2img'][0]) if scene_data['cam2img'] else 0}")
        print(f"cam2global: {len(scene_data['cam2global'])} frames")
        print(f"axis_align_matrix shape: {len(scene_data['axis_align_matrix'])}x{len(scene_data['axis_align_matrix'][0]) if scene_data['axis_align_matrix'] else 0}")

        print("\nElements:")
        for elem in scene_data['elements']:
            match_str = ""
            if elem.get('transform_matches') is not None:
                match_str = " [MATCH]" if elem['transform_matches'] else " [MISMATCH]"

            print(f"  [{elem['index']}] object_id: {elem['metadata'].get('object_id', 'N/A')}{match_str}")
            print(f"       input_box[:3] (box_center): {elem['box_center']}")
            print(f"       transformed_center (from text): {elem['transformed_center']}")
            print(f"       computed_transformed_center:    {elem['computed_transformed_center']}")
            print()

def run_scan2cap(
    scan2cap_annotation_path="data/demo_data/scan2cap_debug_32frames_point3r.json",
    data_dir="data/media",
    output_path="data/demo_data/scan2cap_debug_results.json",
    model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    auto_preprocess=False,
    save_preprocessed=True,
    use_viser=False
):
    """
    Run scan2cap evaluation on a dataset with pre-computed pointer memory.

    Args:
        scan2cap_annotation_path: Path to the scan2cap annotation JSON file
        data_dir: Base directory for resolving pointer_data paths
        output_path: Path to save results JSON file
        model_path: Path to model checkpoint or HuggingFace model ID
                   Examples:
                   - "Qwen/Qwen2.5-VL-3B-Instruct" (base model, default)
                   - "outputs/scan2cap_point3r_all_frames" (fine-tuned)
        auto_preprocess: If True, automatically generate pointer_data when file is missing
                        by loading images from posed_images directory (default: False)
        save_preprocessed: If True, save generated pointer_data to disk for future use
                          Only used when auto_preprocess=True (default: True)
        use_viser: Enable viser visualization during preprocessing (default: False)
    """
    import json
    import os
    from time import time

    # Stage 0: Load annotations
    print("\n" + "="*70)
    print("Scan2Cap Evaluation")
    print("="*70)
    print(f"Model: {model_path}")

    start_time = time()

    # Read scan2cap annotation file
    print(f"\nLoading annotations from: {scan2cap_annotation_path}")
    with open(scan2cap_annotation_path, 'r') as f:
        annotations = json.load(f)

    print(f"Total samples: {len(annotations)}")

    # Count unique scenes
    unique_scenes = set()
    for sample in annotations:
        if 'pointer_data' in sample:
            scene_id = sample['pointer_data'].split('/')[2].replace('.pt', '')
            unique_scenes.add(scene_id)
    print(f"Unique scenes: {len(unique_scenes)}")

    # Stage 1: Load models (without Point3R since using preprocessed data)
    print("\n" + "="*70)
    print("Loading Models")
    print("="*70)
    model_start = time()

    model, processor, min_pixels, max_pixels, _ = load_models(
        load_point3r=False,
        model_path=model_path
    )

    model_end = time()
    print(f"Model loading time: {model_end - model_start:.2f} seconds")

    # Stage 2: Process each sample
    print("\n" + "="*70)
    print("Processing Samples")
    print("="*70)

    results = []
    success_count = 0
    fail_count = 0

    # Cache to avoid reloading the same pointer data file multiple times
    pointer_data_cache = {}

    for idx, sample in enumerate(annotations):
        try:
            # Extract query from conversation
            conversation_value = sample['conversations'][0]['value']
            # Remove special tokens to get clean query
            # Format: "<|vision_start|><|pointer_pad|><|vision_end|>\nActual question here"
            query = conversation_value.split('<|vision_end|>')[-1].strip()

            # Get pointer data path
            pointer_data = sample['pointer_data']
            pointer_data_path = os.path.join(data_dir, pointer_data)
            print(f'pointer_data_path: {pointer_data_path}')

            # Get ground truth
            ground_truth = sample['conversations'][1]['value']

            # Check if pointer data file exists (must be a file, not a directory)
            if not os.path.exists(pointer_data_path):
                if not auto_preprocess:
                    print(f"\n{'='*70}")
                    print(f"Sample {idx+1}/{len(annotations)} - ERROR")
                    print(f"Pointer data file not found: {pointer_data_path}")
                    print("Set auto_preprocess=True to generate it automatically")
                    fail_count += 1
                    continue

                # Auto-preprocessing enabled - generate pointer data
                print(f"\n{'='*70}")
                print(f"Sample {idx+1}/{len(annotations)} - Preprocessing")
                print(f"Pointer data file not found: {pointer_data_path}")
                print("Auto-preprocessing enabled - generating pointer data...")

                # Load Point3R model on-demand (only first time)
                if 'point3r_model' not in locals():
                    print("Loading Point3R model for preprocessing...")
                    point3r_model = Point3R.from_pretrained("./cache/point3r_512.pth")
                    point3r_model = point3r_model.to("cuda")
                    point3r_model.eval()

                # Extract scene_id from pointer_data path
                scene_id = extract_scene_id_from_pointer_path(pointer_data)

                # Construct input images directory
                # pointer_data: "scannet/pointer_memory_debug/scene0000_00.pt"
                # -> input_images_dir: "data_dir/scannet/posed_images/scene0000_00/"
                posed_images_subdir = pointer_data.replace("pointer_memory_debug", "posed_images").replace(".pt", "")
                input_images_dir = os.path.join(data_dir, posed_images_subdir)

                # Construct ScanNet GT pth path for visualization
                # pointer_data: "scannet/pointer_memory_debug/scene0000_00.pt"
                # -> scannet_pth_path: "data_dir/scannet/pcd_with_object_aabbs/val/scene0000_00.pth"
                for task in ['train', 'val', 'test']:
                    scannet_pth_subdir = f"scannet/pcd_with_object_aabbs/val/{scene_id}.pth"
                    scannet_pth_path = os.path.join(data_dir, scannet_pth_subdir)
                    if os.path.isfile(scannet_pth_path):
                        break
                else:
                    raise FileNotFoundError("Scannet GT File not found")
                scannet_pose_subdir = f"scannet/posed_images/{scene_id}/"
                scannet_pose_dir = os.path.join(data_dir, scannet_pose_subdir)

                # Check if images directory exists
                if not os.path.exists(input_images_dir):
                    print(f"Images directory not found: {input_images_dir}")
                    fail_count += 1
                    continue

                # Generate pointer data
                try:
                    # Call preprocess_images with appropriate parameters
                    preprocess_images(
                        model=model,
                        processor=processor,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                        point3r_model=point3r_model,
                        input_images_dir=input_images_dir,
                        input_poses_dir=scannet_pose_dir,
                        pointer_data_path=pointer_data_path,  # None if not saving
                        use_viser=use_viser,
                        unload_point3r_model=False,  # Keep model loaded for subsequent preprocessing
                        annotation_result=extract_box_and_coordinates_from_scan2cap(scan2cap_annotation_path),
                        scannet_pth_path=scannet_pth_path if use_viser else None,
                        image_extensions=("*.jpg",),
                    )

                except Exception as e:
                    print(f"Error during preprocessing: {e}")
                    import traceback
                    traceback.print_exc()
                    fail_count += 1
                    continue

            # Load pointer data with caching
            if pointer_data_path not in pointer_data_cache:
                pointer_data_cache[pointer_data_path] = torch.load(pointer_data_path, weights_only=False)
                print(f"Loaded and cached pointer data from {pointer_data_path}")

            # Run inference with cached pointer data
            generated_response = run_models(
                model=model,
                processor=processor,
                pointer_data_path=pointer_data_path,
                query=query,
                pointer_data=pointer_data_cache[pointer_data_path]
            )

            # Display results
            print(f"\n{'='*70}")
            print(f"Sample {idx+1}/{len(annotations)}")
            print(f"Scene: {sample['metadata'].get('scene_id', 'N/A')}")
            print(f"Object: {sample['metadata'].get('object_id', 'N/A')}")
            print(f"Question Type: {sample['metadata'].get('question_type', 'N/A')}")
            print(f"\nQuery: {query}")
            print(f"\nGenerated: {generated_response}")
            print(f"\nGround Truth: {ground_truth}")

            # Store results
            results.append({
                'sample_id': idx,
                'metadata': sample['metadata'],
                'query': query,
                'generated_response': generated_response,
                'ground_truth': ground_truth,
                'input_box': sample.get('input_box', None),
                'gt_box': sample.get('gt_box', None),
                'iou': sample.get('iou', None)
            })
            success_count += 1

        except Exception as e:
            print(f"\n{'='*70}")
            print(f"Sample {idx+1}/{len(annotations)} - ERROR")
            print(f"Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue

    # Cleanup pointer data cache to free memory
    print(f"\nCleaning up pointer data cache ({len(pointer_data_cache)} entries)...")
    pointer_data_cache.clear()

    # Cleanup Point3R model if it was loaded for preprocessing
    if 'point3r_model' in locals():
        print("Unloading Point3R model...")
        del point3r_model
        torch.cuda.empty_cache()

    # Stage 3: Save results and print summary
    print("\n" + "="*70)
    print("Scan2Cap Evaluation Complete")
    print("="*70)

    end_time = time()

    print(f"Total samples: {len(annotations)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

    # Save results to JSON file
    if output_path:
        print(f"\nSaving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved successfully!")

    print("="*70)

if __name__=='__main__':
    # Example 1: Run scan2cap with base model
    # run_scan2cap()

    # # Example 2: Run scan2cap with fine-tuned checkpoint
    # run_scan2cap(
    #     scan2cap_annotation_path="data/demo_data/scan2cap_debug_32frames_point3r.json",
    #     output_path="data/demo_data/scan2cap_val_output.json",
    #     auto_preprocess=True,
    #     use_viser=True
    #     # model_path="outputs/scan2cap_point3r_all_frames",
    # )

    # Example 3: Preprocess images and run inference (original demo)
    # input_images_dir = "./data/demo_data/sample_data"
    # pointer_data_path = "./data/demo_data/sample_data/pointer_data_qwen3.pt"
    input_images_dir = "./data/media/scannet/posed_images/scene0000_01"
    pointer_data_path = "./data/demo_data/scene0000_01.pt"
    # input_images_dir = "./data/demo_data/arkit_47895700"
    # pointer_data_path = "./data/demo_data/arkit_47895700.pt"
    
    # query = "Describe this image."
    query = "Describe this scene in detail, in the order of appearances."
    model_path="Qwen/Qwen3-VL-4B-Instruct"
    use_viser = False
    sample_ct = 32
    model, processor, min_pixels, max_pixels, point3r_model = load_models(model_path=model_path)
    # preprocess_images(model, processor, min_pixels, max_pixels, point3r_model,
    #                   input_images_dir, pointer_data_path, use_viser, sample_ct=sample_ct, max_memory_tokens=None, 
    #                   image_extensions = ("*.jpg", ))
    run_models(model, processor, pointer_data_path, query)