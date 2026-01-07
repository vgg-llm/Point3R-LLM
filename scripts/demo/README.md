# Demo Scripts for Point3R-LLM

This folder contains demonstration scripts for using Qwen2.5-VL models with and without Point3R memory integration.

## Files

### 1. `demo.py` - Standard Qwen2.5-VL Demo
A simple demo using the standard Qwen2.5-VL model for image understanding tasks.

**Features:**
- Memory-efficient loading with bf16 precision
- Automatic device mapping
- Uses standard `Qwen2_5_VLForConditionalGeneration`

**Usage:**
```bash
python scripts/demo/demo.py
```

**What it does:**
- Loads the Qwen2.5-VL-3B model
- Processes an image URL with a text question
- Generates a response describing the image

---

### 2. `demo_point3r.py` - Point3R-Enhanced Demo
Demonstrates how to use the Point3R-enhanced model with pointer memory support.

**Features:**
- Uses `Qwen2_5_VLForConditionalGenerationWithPoint3R`
- Uses `Qwen2_5_VLProcessorWithPoint3R`
- Supports pointer tokens for 3D scene memory
- Shows both standard and pointer-enhanced usage
- Extracts actual Point3R memory from images

**Usage:**
```bash
python scripts/demo/demo_point3r.py
```

**What it does:**
1. **Example 1:** Standard image processing (without pointers)
2. **Example 2:** Extracts Point3R memory from images and generates responses using pointer tokens

---

### 3. `extract_pointer_memory.py` - Pointer Memory Extraction Utility
Utility module for extracting Point3R memory features from images.

**Key Function:**
```python
extract_pointer_memory(
    image_inputs,           # List of images (paths, PIL, or numpy)
    point3r_model,          # Loaded Point3R model
    device='cuda',          # Device to use
    no_crop=False,          # Whether to resize instead of crop
    size=512,               # Target image size
    verbose=True            # Print progress info
)
```

**Returns:**
- `pointer_memory_embeds`: Memory embeddings for pointer tokens
- `pointer_positions`: 3D positions (height, width, depth)
- `pts3d`: Raw 3D point cloud
- `metadata`: Processing metadata

**Usage:**
```python
from extract_pointer_memory import extract_pointer_memory
from qwen_vl.model.point3r.point3r import Point3R

# Load Point3R model
point3r = Point3R.from_pretrained("path/to/checkpoint.pth")
point3r = point3r.to('cuda').eval()

# Extract memory
pointer_data = extract_pointer_memory(
    image_inputs=['image1.jpg', 'image2.jpg'],
    point3r_model=point3r,
)

# Use with model
outputs = model.generate(
    **inputs,
    pointer_memory_embeds=pointer_data['pointer_memory_embeds'],
    pointer_positions=pointer_data['pointer_positions'],
)
```

---

## Key Differences Between Models

### Standard Model (`Qwen2_5_VLForConditionalGeneration`)
- Standard vision-language model
- Supports images and videos
- Uses 3D RoPE (Rotary Position Embeddings) for vision tokens

### Point3R-Enhanced Model (`Qwen2_5_VLForConditionalGenerationWithPoint3R`)
- Extends the standard model
- **Additional feature:** Supports pointer memory tokens from Point3R
- Uses **4D RoPE** for pointer tokens (temporal, height, width, depth)
- Pointer tokens encode 3D spatial information from scenes

### Standard Processor (`Qwen2_5_VLProcessor`)
- Processes images, videos, and text
- Uses standard special tokens: `<|image_pad|>`, `<|video_pad|>`

### Point3R-Enhanced Processor (`Qwen2_5_VLProcessorWithPoint3R`)
- Extends the standard processor
- **Additional feature:** Adds `<|pointer_pad|>` token
- Can process `LocalMemory` objects from Point3R
- Handles pointer token insertion and positioning

---

## Using Pointer Memory (Advanced)

To use pointer memory, you need:

1. **Point3R Memory Features:** Extract features from a 3D scene using Point3R
2. **LocalMemory Object:** Create a `LocalMemory` object containing the features
3. **Pointer Positions:** Provide 3D coordinates (height, width, depth) for each pointer

Example structure:
```python
from qwen_vl.model.point3r.point3r import LocalMemory

# Extract Point3R memory from your 3D scene
pointer_memory = LocalMemory(...)  # Your implementation

# Define pointer positions in 3D space
pointer_positions = torch.tensor([
    [h1, w1, d1],  # First pointer position
    [h2, w2, d2],  # Second pointer position
])

# Process with pointer memory
inputs = processor(
    text=["<|pointer_pad|> What objects are in this scene?"],
    pointers=pointer_memory,
    return_tensors="pt",
)

# Generate with pointer support
outputs = model.generate(
    **inputs,
    pointer_positions=pointer_positions,
    max_new_tokens=128,
)
```

---

## Memory Optimization

Both demos use several memory optimization techniques:

1. **`torch_dtype=torch.bfloat16`** - Uses 16-bit precision (50% less memory)
2. **`device_map="auto"`** - Automatically distributes model across GPUs
3. **`low_cpu_mem_usage=True`** - Reduces CPU memory during loading
4. **`torch.inference_mode()`** - Disables gradients for inference
5. **`do_sample=False`** - Uses greedy decoding (faster, less memory)

If you still encounter OOM errors:
- Reduce `max_new_tokens`
- Use a smaller model variant
- Use CPU offloading: `device_map={"": "cpu"}` (slower but won't OOM)

---

## Requirements

```bash
pip install torch transformers accelerate qwen-vl-utils
```

For Point3R features, you'll also need the Point3R dependencies (see main README).

---

## Related Files

- Model implementation: [src/qwen_vl/model/modeling_qwen_point3r.py](../../src/qwen_vl/model/modeling_qwen_point3r.py)
- Processor implementation: [src/qwen_vl/model/processing_qwen2_5_vl.py](../../src/qwen_vl/model/processing_qwen2_5_vl.py)
- Test script: [test_pointer_rope.py](../../test_pointer_rope.py)
