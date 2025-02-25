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

Learn about using hugging face to aqquire data, how to prepare such data for training, selecting a fast but accurate detection model thats pretrained for finetuning, and tweaking hyperparameters to optmise the perofmrance and accuracy of training so you can go from idea to generalised solution on detection inference.

We specifically showcase training on the manufacterer type column of the FGVC-Aircraft dataset.
This is more tailored to EveryID, because the ideal thing is to not just detect humans but to detect types of humans, or in this case the specific palne manufactuer and not just that we have a plane.
This will mean when ou run EveryPerson through footage, the top_k results dont have silly predicitions after the top10_k.
This might not be a problem for top rank , but it certianly is for mean average precision (mAP) which is the most meaningful reid metric for generalisaiton. At postprocessing the crucial stage of clustering indviduals means we need high mean average precision for each person , making clusters a viable post processing method to signifcantly enhance the perfofmance of reid.

Even to be able to narrow down whether someone is young, old, male or female will meaninfully increase the mAP of reid, while being completely detached from that process , which is clean. A kind of cross examination is certainly what's required, and if there is as much metadata as possible associated with peoples then impossible matches are prevented , thus improving clustering through higher precision, without any actual improvment of the reid model itself. 

A lot of the clever solutions throughout the constrcution of this project will certainly come from extracting all value out of every pre and post prcoessing step. This is what many researchers have failed to understand is the fundamental problem.

If someone is in the same clothes and easy to see, reid should and is simple enough, but no practical reid use case is without immense noise beyond what vectors can reconcile to reason. Therefore solutions come from understanding what represents a person, because a representation has multiple dimensions. Similairty is certianly one of those, but not the only dimension.

I am reappraising the problem with fresh eyes as i made all these mistakes last year.

ideally use a gpu for training, but it will work with cpu , just be slower. This is a relatively small set and model, so both will work.

```bash
python object/plane_manufacturer_detection_cookbook/train.py
```