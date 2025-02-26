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

## Vision

The goal is to push transformers beyond current benchmarks for recognition tasks. We've achieved this on Market1501, the hallmark person ReID benchmark, scoring above 98% on top rank accuracy.

Recognition is more complex than detection - it requires understanding invariant representations of entities that change over time. Our goal is to give machines this same capability that humans possess naturally.

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

6. **Hierarchical Recognition**: Implementing a multi-stage approach that progressively refines identification.

## Setup

```bash
git clone https://github.com/cookieclicker123/EveryID
cd everyid
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Object Detection

The object detection pipeline is a prerequisite for person ReID, identifying all people in frames before recognition occurs.

### Architecture
- Two-stage approach: YOLOv8 for detection + SAM for segmentation
- Multi-class detection (80+ classes including people)
- Instance segmentation and tracking capabilities

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

### Usage

```bash
# Run scene analysis tests
python3 run.py scene tests

# Download the scene CNN model
python3 run.py scene download_cnn

# Download the scene transformer model
python3 run.py scene download_transformer

# Run scene analysis inference with CNN model
python3 run.py scene inference_cnn

# Run scene analysis inference with transformer model
python3 run.py scene inference_transformer
```

### Testing the Model

Add test images to `./tmp/test_scene/` and run:
```bash
python3 run.py scene inference_transformer
```
Results will be saved to `./tmp/scene_results/`.