# SAM 2: Segment Anything in Images and Videos

**Authors:** Meta AI (FAIR)
**Published:** August 2024
**Paper:** [arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)

---

## Why This Matters

SAM 2 brought **universal object segmentation to video**:

- 🎬 **Segment anything in video** - Any object, real-time tracking
- 🚀 **6× faster than SAM 1** - Improved architecture
- 🎯 **Zero-shot video segmentation** - No training on target domain
- 🌍 **Open source** - Apache 2.0, democratizes video AI
- 🔧 **Promptable** - Click, box, or text to segment

**Real-world impact:**
- Video editing (automatic object removal/tracking)
- Autonomous vehicles (object tracking)
- Medical imaging (organ segmentation in video)
- AR/VR (real-time segmentation)

**The insight:** **Unified model for images AND video** - not separate models, one architecture handles both.

---

## Key Innovations

### 1. Unified Image-Video Architecture

**Previous approach:**
```
Image segmentation: SAM 1
Video segmentation: Separate models
Problem: Inconsistent across frames
```

**SAM 2 approach:**
```
One model handles:
- Single images (like SAM 1)
- Video sequences (NEW)
- Maintains consistency across frames
```

### 2. Streaming Architecture

**Memory mechanism:**
```
Process video frame-by-frame:
Frame 1: Segment object, store in memory
Frame 2: Use memory to find object, update
Frame 3: Use memory, propagate mask
...

Benefits:
- Real-time processing
- Handles occlusions
- Recovers from errors
```

### 3. Promptable Segmentation

**How to use:**
```
Method 1: Click on object (in any frame)
→ SAM 2 tracks through entire video

Method 2: Draw bounding box
→ Segments and tracks

Method 3: Text prompt (with future models)
→ "Segment the car"
```

---

## Performance

### Video Object Segmentation

**SA-V (Segment Anything Video benchmark):**

| Model | J&F Score |
|-------|-----------|
| XMem | 72.3 |
| Cutie | 76.8 |
| **SAM 2** | **82.5** |

**7% absolute improvement!**

### Image Segmentation

**SA-1B (Original SAM benchmark):**
- SAM 1: 85.3 IoU
- **SAM 2: 87.1 IoU** (better than original!)

**Improves on images too!**

### Speed

**Frames per second:**
```
SAM 1 on video: ~4 FPS (slow)
SAM 2: ~44 FPS (real-time!)

11× faster than baseline
6× faster than SAM 1
```

---

## Architecture

### Streaming Memory

```
Current frame
    ↓
Image encoder (ViT-H)
    ↓
Prompt encoder (clicks, boxes)
    ↓
Memory attention (look at past frames)
    ↓
Mask decoder
    ↓
Update memory bank
```

**Memory mechanism:**
```python
# Simplified
class SAM2Memory:
    def __init__(self):
        self.memory_bank = []  # Store past frames
        
    def process_frame(self, frame, prompt=None):
        # Encode current frame
        features = self.image_encoder(frame)
        
        # Attend to memory (past frames)
        context = self.memory_attention(features, self.memory_bank)
        
        # Generate mask
        mask = self.mask_decoder(features, context, prompt)
        
        # Update memory
        self.memory_bank.append({
            'frame': frame,
            'mask': mask,
            'features': features
        })
        
        return mask
```

---

## Practical Usage

### Basic Image Segmentation

```python
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# Load image
image = load_image("photo.jpg")
predictor.set_image(image)

# Segment with point prompt
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),  # Click coordinates
    point_labels=np.array([1]),  # 1 = foreground
)

# Get best mask
best_mask = masks[scores.argmax()]
```

### Video Segmentation

```python
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Load model
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

# Initialize with video
predictor.init_state(video_path="video.mp4")

# Add prompt in first frame
frame_idx = 0
obj_id = 1
predictor.add_new_points(
    frame_idx=frame_idx,
    obj_id=obj_id,
    points=np.array([[200, 300]]),
    labels=np.array([1])
)

# Propagate through video
for frame_idx, masks in predictor.propagate_in_video():
    # masks[obj_id] contains segmentation for this frame
    save_mask(masks[obj_id], f"mask_{frame_idx}.png")
```

### Interactive Refinement

```python
# User clicks to add/remove regions
predictor.add_new_points(
    frame_idx=10,
    obj_id=1,
    points=np.array([[x1, y1]]),  # Add this region
    labels=np.array([1])
)

predictor.add_new_points(
    frame_idx=10,
    obj_id=1,
    points=np.array([[x2, y2]]),  # Remove this region
    labels=np.array([0])  # 0 = background
)

# Re-propagate with corrections
for frame_idx, masks in predictor.propagate_from_frame(10):
    # Updated masks incorporating user feedback
    pass
```

---

## Real-World Applications

### 1. Video Editing

```python
# Remove object from video
# 1. Segment object with SAM 2
# 2. Use inpainting to fill
# 3. Consistent across all frames

def remove_object_from_video(video_path, click_point):
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
    predictor.init_state(video_path)
    
    # Segment object
    predictor.add_new_points(0, 1, [click_point], [1])
    
    # Get masks for all frames
    masks = {}
    for frame_idx, frame_masks in predictor.propagate_in_video():
        masks[frame_idx] = frame_masks[1]
    
    # Inpaint each frame (using your inpainting model)
    for frame_idx, mask in masks.items():
        frame = load_frame(video_path, frame_idx)
        inpainted = inpaint(frame, mask)
        save_frame(inpainted, frame_idx)
```

### 2. Object Tracking

```python
# Track person through video
person_tracker = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
person_tracker.init_state(surveillance_video)

# Click on person in frame 0
person_tracker.add_new_points(0, person_id=1, [[x, y]], [1])

# Track through video
bounding_boxes = []
for frame_idx, masks in person_tracker.propagate_in_video():
    mask = masks[1]
    bbox = get_bounding_box(mask)
    bounding_boxes.append(bbox)
    
# Now have full trajectory
```

### 3. Medical Imaging

```python
# Segment organ in medical video (ultrasound, etc.)
medical_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
medical_predictor.init_state(ultrasound_video)

# Doctor clicks on organ
medical_predictor.add_new_points(0, organ_id=1, doctor_click, [1])

# Automatic segmentation through all frames
for frame_idx, masks in medical_predictor.propagate_in_video():
    organ_mask = masks[1]
    compute_volume(organ_mask)  # Track organ size over time
```

---

## Key Takeaways

1. **Unified image-video model** - One architecture for both
2. **Real-time video segmentation** - 44 FPS
3. **Zero-shot** - Works on any domain without retraining
4. **Interactive** - User can correct mistakes
5. **Open source** - Freely available

**Bottom line:** SAM 2 made universal video segmentation practical and accessible, opening new possibilities for video editing, tracking, and analysis.

---

## Further Reading

- **Paper:** https://arxiv.org/abs/2408.00714
- **Code:** https://github.com/facebookresearch/segment-anything-2
- **Demo:** https://sam2.metademolab.com/

**Published:** August 2024
**Impact:** 🔥🔥🔥🔥 **HIGH** - Universal video segmentation
**Adoption:** Widespread in video editing tools
