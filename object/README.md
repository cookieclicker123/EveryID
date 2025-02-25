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

## Train a detection model from scratch on a bad use case

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

## Train a detection model from scratch on a GOOD use case

Now that you've seen where object detection is inhrenetly limited , lets show where its not.

Intra class problems its always going to struggle on, because you are going to have way more classes , and most likely insufficient images per class, as well as imbalances. these are the things that slow training and prevent convergence through simply not enough infromation to the network, and not quantity of data, because we have more data in the dataset we are using here yet convergence is substantially quicker as you can see.

We are going to now showcase where yolo can be a beast for you : the CIFAR-10 dataset, and we train on just boats, automobiles, trucks and planes. 5000 training images and 1000 val images per class.

We even use yolov8n.pt as opposed to yolo11l.pt just to showcase an inferior detector can solve a problem on an amenable deteciton problem than a state of the art detector on a porrly set up problem like before.

First confirm you have the dataset and expose its metadata and annotations info:

```bash
python object/vehicle_classifier/download_test.py
python object/vehicle_classifier/annotations_test.py
```

Once confirmed, commit to 20 epochs of training:

```bash
python object/vehicle_classifier/train.py
```

As you can see, both performance(training time per epoch) and accuracy(mean average precision(mAP)) are great on this run. This proves to us that this problem was machine learning amenable for detection models.

we even introduce the disadvantage of using a dataset with super low resolution images, and an inferior model , just to showcase the importance of the class_number/images_per_class. in addition , having an distinct, intercalss problem helps too. the previous problem probably had a lot of very similiar types, so to get that working would require massive amounts of data, which simply isnt feasible.

This should convey to you that object detection is a crucial stage of the EveryPerson pipeline, but not the reid itself.

REID requires purpose built models, and fundamentally different engineering solutions. If it didnt work on stanford cars then imagine the pain youd have trying to do this on openset people. It simply wouldnt work.

What we will explore in dino folder however is we can constrain possibilities in the detection phase of processing in order to help reid. for example, if we could move to open set detectors, we should be able to weed out silly matches in the reid model, e.g wrong gender, age, place and time etc.

Although the top rank accuracy of our model is very impressive as you will see and defeats the benchmark, to attain high mean average precision and make clustering actually work in post processing will require methods outside of the control of the reid model which purely models pixel wise similarity.

test the inference script on some images of each class, perhpas four simple images from elsewhere on each category to see if generalsiation has occured:

```bash
python object/vehicle_classifier/inference_test.py
```
