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
base_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

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

print(f"Pointer token: {processor.pointer_token}")
print(f"Pointer token ID: {processor.pointer_token_id}")

# Example 1: Using the model with standard image input (no pointers)
print("\n" + "="*70)
print("Example 1: Standard Image Input (without pointers)")
print("="*70)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "What is the type of the Dog?"},
        ],
    }
]

# Process inputs without pointers
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
from qwen_vl_utils import process_vision_info
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Move inputs to the same device as the model
inputs = inputs.to(model.device)

# Generate with memory-efficient settings
print("Generating response...")
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,  # Greedy decoding to save memory
    )

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"Response: {output_text[0]}")

# Example 2: Using the model with pointer memory (demonstration)
print("\n" + "="*70)
print("Example 2: With Pointer Memory (demonstration)")
print("="*70)
print("Note: This requires actual Point3R memory features from a 3D scene.")
print("This example shows the structure - you would need to:")
print("  1. Load a 3D scene and extract Point3R features")
print("  2. Create LocalMemory object with those features")
print("  3. Pass it to the processor as 'pointers' parameter")
print()
print("Example code structure:")
print("""
# Hypothetical usage with Point3R memory:
from qwen_vl.model.point3r.point3r import LocalMemory

# 1. Extract Point3R memory from a 3D scene (implementation-specific)
# pointer_memory = extract_point3r_memory(scene_data)

# 2. Create message with pointer token
messages_with_pointer = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Based on the 3D scene memory: <|pointer_pad|>"},
            {"type": "text", "text": "What objects are present?"},
        ],
    }
]

# 3. Process with pointer memory
text = processor.apply_chat_template(messages_with_pointer, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    pointers=pointer_memory,  # Pass LocalMemory object here
    return_tensors="pt",
)

# 4. You also need to provide pointer_positions (height, width, depth)
pointer_positions = torch.tensor([
    [h1, w1, d1],  # Position for first pointer token
    [h2, w2, d2],  # Position for second pointer token
    # ... etc
])

# 5. Generate with pointer inputs
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        pointer_positions=pointer_positions,
        max_new_tokens=128,
    )
""")

print("\n" + "="*70)
print("Demo completed!")
print("="*70)
