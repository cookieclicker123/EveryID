# EveryID: Recognition of People, Objects, Scenes and Events

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/)
[![transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-pink)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pydantic](https://img.shields.io/badge/pydantic-v2.5-ff3399.svg)](https://docs.pydantic.dev/)
[![Typing](https://img.shields.io/badge/Typing-Supported-brightgreen.svg)](https://docs.python.org/3/library/typing.html)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Pillow](https://img.shields.io/badge/Pillow-8.0%2B-brightgreen.svg)](https://python-pillow.org/)


![EveryID](img/EveryID.jpg)

## Table of Contents
- [Vision](#vision)
- [Applications](#applications)
- [Fundamental Challenges](#fundamental-challenges)
- [Proposed Solutions](#proposed-solutions)
- [Setup](#setup)
- [Object Detection](#object-detection)
- [Person ReID](#person-reid)
- [Scene Recognition](#scene-recognition)
- [Future Directions](#future-directions)
- [Supporting Research Areas](#supporting-research-areas)


## Vision for this specific Repository

The goal is to push transformers beyond current benchmarks for recognition tasks. We've achieved this on Market1501, the hallmark person ReID benchmark, scoring above 98% on top rank accuracy.

Recognition is more complex than detection - it requires understanding invariant representations of entities that change over time. Our goal is to give machines this same capability that humans possess naturally.

## Mission Statement

To understand the full mission statement and the fundamental reasons behind starting this company, please visit [www.everyid.ai](http://www.everyid.ai).


## Applications

### TV Post Production
- Automatic logging and syncing across multiple footage sources
- Instant searchability based on people, scenes, and concepts
- Knowledge graph generation for content clusters
- Context-aware tracking with probabilistic likelihood
- Real-time indexing of large productions

### Surveillance
- Track individuals across non-overlapping camera networks
- Maintain identity consistency despite appearance changes
- Handle crowded scenes and occlusions
- Process real-time video streams efficiently
- Scale to large camera networks

### Government Intelligence
- Rapid processing of surveillance footage
- Cross-referencing across multiple data sources
- Temporal analysis of movement patterns
- Integration with existing intelligence systems
- Handling of low-quality imagery

### Agent-Based AI Frameworks
- Moving from retrieval to recognition paradigms
- Enabling prediction of future events through pattern recognition
- Building agents that can adapt plans dynamically
- Creating self-reinforcement from failure

## Fundamental Challenges

1. **Intra-class vs Inter-class Complexity**: Person ReID is an intra-class problem (distinguishing between instances of the same class) which is inherently more difficult than inter-class problems like object detection.

2. **Appearance Variability**: People change appearance across time and cameras (clothing, pose, lighting, viewpoint).

3. **Occlusion and Partial Views**: Often only partial information is available in crowded scenes.

4. **Camera Variations**: Different cameras have different characteristics (resolution, color balance, angle).

5. **Temporal Consistency**: Maintaining identity across time requires more than frame-by-frame matching.

6. **Scalability**: Systems must handle thousands of identities efficiently.

## Proposed Solutions

1. **Advanced Transformer Architectures**: Utilizing state-of-the-art vision transformers trained on large datasets to extract robust features.

2. **Multi-modal Fusion**: Combining appearance, temporal, and contextual information for more reliable identification.

3. **Temporal Modeling**: Incorporating time as a dimension in the recognition process rather than treating frames independently.

4. **Scene Context Integration**: Using scene understanding to improve person ReID by constraining the search space.

5. **Self-supervised Learning**: Leveraging unlabeled data to improve generalization through techniques like DINO grounding.

6. **Clustering and reranking**: Clustering similar tracks and reranking them to improve the essential mAP metric through.

7. **Hierarchical Recognition**: Implementing a multi-stage approach that progressively refines identification.

## Setup

```bash
git clone https://github.com/cookieclicker123/EveryID
cd everyid
pip install --upgrade pip
```

## Object Detection

The object detection pipeline is a prerequisite for person ReID, identifying all people in frames before recognition occurs.

### Architecture
- 3-stage approach: YOLOv8n for accurate , lightweight detection + DINO for open set detection + bytetrack for tracking

### Usage
```bash
# Run object detection tests
python3 run.py object tests

# Run object detection on a sample image
python3 run.py object detection

# Run object detection on a sample video
python3 run.py object video

# Run object tracking on a sample video
python3 run.py object tracking
```

## Training an Object Detector

In `object/car_classifier`, we demonstrate a challenging problem where detection is insufficient: classifying cars by type, make, and model. This is an **intra-class problem**, where the goal shifts from detecting objects to differentiating between subtle variations within the same class. 

### Key Challenges:
- **Complexity:** Classifying car types requires high-resolution data, significant computational resources, and large models like `yolo11l.pt` trained on datasets such as the **Stanford Cars Dataset**.
- **Accessibility:** Without extensive data per class and powerful compute resources, this problem is difficult for most solo developers.

In contrast, in `vehicle_classifier`, we showcase an **inter-class problem** where categories are distinct (e.g., ships, planes, trucks, and cars). Here, detection alone suffices. Using a small pre-trained model like `yolov8n.pt` (3 million parameters) trained on low-resolution images from **CIFAR-10**, we demonstrate:
- **Efficiency:** Even with deliberately handicapped conditions, detection is effective for inter-class problems.
- **Speed:** Small models allow ultra-fast training and inference.

### Lessons Learned:
1. **Don’t Overengineer:** Use small models when the problem is simple. Detection is a critical stage in pipelines like Person ReID and can often meet requirements without large models.
2. **Detection vs Classification:**
   - Detection works well for inter-class problems with distinct categories and sufficient data per class.
   - Classification becomes essential for intra-class problems where subtle differences within the same category must be identified.

---

## Deep Learning Examples in `object/`
These folders demonstrate how to train detection models effectively while highlighting the limitations of basic detection for Person ReID:

1. **Why Detection Alone Falls Short for Person ReID:**
   - Even with accurate person detection, we need more than inter-class detection to solve Person ReID problems.
   - Open-set detection models like DINO offer a more robust solution by capturing richer metadata beyond simple bounding boxes.

2. **Intra-Class Challenges:**
   - As seen with `car_classifier`, intra-class classification is not easily achievable using standard detection models like YOLO.

3. **Inter-Class Simplicity:**
   - The `vehicle_classifier` example highlights how distinct categories and adequate data per class allow small models to perform well.

---

## The Power of Open-Set Detection Models
Open-set models like DINO are pivotal for advancing Person ReID:
- They go beyond basic detection by identifying nuanced characteristics within detected objects.
- This capability is critical when dealing with intra-class challenges or when metadata (e.g., age, gender) is required to improve downstream tasks.

For more details on open-set detection and its applications in Person ReID, refer to the documentation in [`dino/README.md`](dino/README.md).

## Person ReID

### Training a Model

See `person/transformer_msmt17.ipynb` for training a state-of-the-art person ReID model.

To download the MSMT17 dataset:
```bash
mkdir -p ./tmp/datasets && pip install gdown && gdown --id 1nqDWKIbYbnj03HikgzywmkSk9MuCU11Y -O ./tmp/datasets/MSMT17_combined.zip && unzip ./tmp/datasets/MSMT17_combined.zip -d ./tmp/ && rm ./tmp/datasets/MSMT17_combined.zip
```

### Uploading/Downloading Models

When using Hugging Face models, use "SebLogsdon" as the account name.

```bash
# Upload a model
python person/upload_download_models/upload_person_transformer.py

# Download the person transformer model
python3 run.py person download_model

# Run EveryID person recognition
python3 run.py person everyid
```

## Scene Recognition

Scene recognition helps constrain the search space for person ReID by understanding context.

### Training a Model

See `scene/best_scene_classifier.ipynb` for training a scene recognition model.

To download the dataset:
```bash
mkdir -p ./tmp/datasets && pip install gdown && gdown --id 10wleF-tFtCpZIcvHelYbmVy8sozBHmOj -O ./tmp/datasets/places205_reduced_improved.zip && unzip ./tmp/datasets/places205_reduced_improved.zip -d ./tmp/ && rm ./tmp/datasets/places205_reduced_improved.zip
```

Upload a model outside of venv if you want to do that

```bash
python person/upload_download_models/upload_scene_transformer_transformer.py
```

### Usage

```bash
# Run scene analysis tests
python3 run.py scene tests

# Download the scene transformer model
python3 run.py scene download_transformer

# Run scene analysis inference with transformer model
python3 run.py scene inference_transformer
```

As you will see, the models fail because in the real world there are so many more 'scenes' than 16 discrete scene classes, but its impressive that in theory we were able to get this itraclass recongition problem working, but now we should move to other methods.

# Future Directions

## Leveraging Local Vision LLMs
We aim to explore the use of local, non-API-based vision LLMs like LLAVA for scene analysis. Depending on the application, snapshots can be taken at key moments:

- **Dynamic Scenes (e.g., moving camera or edited footage):** Snapshots can be captured intermittently during scene transitions.
- **Static Scenes (e.g., fixed security cameras):** The focus will be on recording "salient events" in real-time, determined by the vision LLM. This process can be enhanced with open-set models that trigger snapshots based on specific thresholds (e.g., velocity changes), enabling the vision LLM to retrospectively analyze significant footage and index it for natural language search.

This approach has immense potential for:
1. **Searchable Video Indexing:** Allowing users to find people, objects, and events via natural language queries across large datasets.
2. **Security Applications:** Offering lightweight, local solutions for security companies, potentially disrupting the market with cost-effective alternatives.

---

## Key Research Areas

### 1. Speed and Latency
- **Feasibility:** Can a local vision LLM like LLAVA operate efficiently in real-time?
- **Snapshot Frequency:** What is the optimal periodicity for capturing snapshots? Sparse snapshots may improve speed but could risk losing critical details in high-pressure scenarios. This might necessitate using lightweight open-set detectors to ensure no important events are missed between snapshots.
- **Performance Metrics:** We will measure LLAVA’s processing speed per frame under different conditions:
  - Optimized with MPS support
  - Running on GPUs like A100 and H100
  - Benchmarked over an hour of edited footage

### 2. Data Model Compliance
- Can LLAVA condense its detailed responses into structured outputs? For example:
  - Convert its analysis into a concise 3-word summary or a single-word classification (e.g., `BEACH`) based on predefined Enum values.
  - Achieving this would combine the generality of LLAVA with the determinism of custom deep learning models, offering both flexibility and reliability.

### 3. Enhancing Person Re-Identification (ReID)
While not essential, LLAVA could assist in narrowing predictions for Person ReID by leveraging its rich metadata:
- Isolating who an individual cannot be, improving the mean Average Precision (mAP) metric.
- Addressing challenges like appearance changes across time (e.g., TV shows or long-term CCTV footage).

---

## Supporting Research Areas

### Open Set Object Detection and Clustering
We will explore methods to enhance Person ReID beyond model training improvements:

1. **Object Detection Phase:**
   - Investigate tools like [DINO Grounding](dino/README.md), an open-set object detector capable of identifying more than just `person`.
   - Attach metadata such as age and gender to detections to eliminate false positives and improve mAP through post-processing.

2. **Post-Processing Phase:**
   - Leverage clustering techniques (e.g., HDBSCAN) to group similar tracks of individuals based on high mAP scores.
   - Establish robust base representations for individuals early in processing to enable self-reinforcement over time as more data is collected.

For further details, refer to:
- [`dino/README.md`](dino/README.md) for object detection research.
- [`clustering/README.md`](clustering/README.md) for clustering methodologies.

---

This roadmap outlines our plans to integrate vision LLMs like LLAVA into scene analysis workflows while addressing key challenges such as speed, accuracy, and scalability.
