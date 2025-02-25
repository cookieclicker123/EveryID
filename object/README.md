# Object detection + tracking cookbooks


## Cookbooks include:


## Why this is relavent to EveryID


## What a focus on the detection phase could potentially solve

## Run the tests

```bash
pytest object/tests/test_object_detection.py -v
pytest object/tests/test_video_object_detection.py -v
pytest object/tests/test_video_object_detection_with_tracking.py -v
```

## Run the object detection workflows

see the stages of increasing complexity, from basic image detection, to class specific tracked footage
with visualisation and CSV metadata.

```bash
python object/img_object_detection.py
python object/video_object_detection.py
python object/video_object_detection_with_tracking.py
```

## Train a detection model from scratch

Learn about using YOLO to train a car classifier from scratch, demonstrating crucial lessons about data requirements and class complexity in real-world detection tasks.

```bash
python object/car_classifier/download_test.py
python object/car_classifier/annotation_test.py
python object/car_classifier/train.py
```

### Key Lessons Learned:

1. Class Complexity vs Data Requirements
   - Initial attempt: 196 car types with ~27 images per class
   - Result: Poor mAP (< 0.0001) despite decreasing loss
   - Lesson: Deep learning needs substantial data per class (typically hundreds, not dozens)

2. Why This Matters for EveryID
   - Person ReID faces similar challenges: many distinct identities (classes) with limited samples
   - Just as cars have subtypes (makes/models), people have attributes (age, gender, clothing)
   - The "closed set" nature of car classification mirrors ReID challenges
   
3. Data Requirements for Reliable Classification
   - Need balance between:
     - Number of classes (complexity)
     - Images per class (representation)
     - Class distinctiveness (feature separation)
   
4. Implications for Person ReID
   - Pure similarity matching isn't enough
   - Need hierarchical attributes (like car make→model)
   - Better to have fewer, well-represented classes than many sparse ones
   - Cross-validation with attributes can improve precision

This experiment demonstrates why EveryID needs:
1. Sufficient examples per identity
2. Hierarchical attribute classification
3. Balance between granularity and generalization
4. Strong per-class representation

The car classification challenge mirrors ReID's core problem: balancing specificity with generalization while managing limited data per class.

