# EveryID: a repository for the recognition of people, objects, scenes and events.

![EveryID](img/EveryID.jpg)

[![Python 3.11](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/)
[![transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-pink)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pydantic](https://img.shields.io/badge/pydantic-v2.5-ff3399.svg)](https://docs.pydantic.dev/)
[![Typing](https://img.shields.io/badge/Typing-Supported-brightgreen.svg)](https://docs.python.org/3/library/typing.html)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Pillow](https://img.shields.io/badge/Pillow-8.0%2B-brightgreen.svg)](https://python-pillow.org/)

The goal is to push the best transformers in the world to perform beyond all current benchmarks for recognition tasks.

One will find their is much more to such a task than simply training a transformer on a large dataset of images and labels.

How does one manage multiple cameras, different footage days, occlusions of light and object, and higher pixel wise similiarity between instances of different 

people than themself? Of course no matter how good the vectors that represent a person are, you can't match everything when they concept of similiarty 
doesnt have such a clear metric of accuracy. People , places, concepts, these are far noisier than simply object detection.

Intra class problems are far more complex than inter class problems in deep learning and this is a continuation of that pattern, which is why 
person reidentification is a largely unsolved problem as opposed to object , ananomly and shape detection.

In a similiar vein, the human visual cortex has to make rapid assumptions about what they've seen and whether its been seen before.

This isnt simple retrieval, this is complex recognition of a permenant representation of a thing that is permanenlty changing.

paradoxial, yet invariant reprsentation in the human brain allows for such a gift.

Our goal is to give machines the same gift.

## Applications

 - TV Post Production: people who log and sync up time code have the laborious job of fishing through thousands of hours of footage in a manual fahsion.
   We should be able to simply index large amounts of footage in a similair way to textual rag systems, and build knowledge graphs of person, scene, and concept clusters that focus more on what is probabilistically likely given context and trakcing , rather than what achieves the most probability frame to frame.
   This way, large productions become immediately searchable and the need for a human editor to watch hours of footage becomes obsolete.

 - Surveillance: 

 - Government intelligence and defence:

 - Revolution in agent based AI frameworks:


## Fundamental challenges:

## How i propose to solve these challenges:

## Initial setup

```bash
git clone https://github.com/cookieclicker123/EveryID?tab=readme-ov-file
cd everyid
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
touch tmp

```

## Person REID

### Training a model

You can observe person/transformer_msmt17.ipynb as an example of how to train a state of the art person reid model.

You are probably going to want GPU resources to train this model as i did.

You can freely customise and upgrade this crucial file and test it on top rank and see if you can get better scores than me.

### Uploading a model to the hub

Ensure you export your hugging face token in your env.

If you are logged in to the hugging face cli, unset any hub keys and logout.

Then log back in, and your token MUST be a write token from the account you want to upload to, not finegrained or read tokens.

```bash
unset HF_TOKEN
huggingface-cli logout
huggingface-cli login
```

Upload a model if you've used the framework to train a model

```bash
python3 upload_download_models/upload_person_transformer.py
```

### Download a model if you want to use the  pre-trained person REID model.

Follow the same steps as uploading a model, but download the model with SebLogsdon as your username.

```bash
python3 upload_download_models/download_person_transformer.py
```

### Test the model

Once you have the model downloaded, you can test it on top rank accuracy on a small subset of images and verify the name_folders match the top rank names.

```bash
python3 person/EveryID_msmt17_top_rank.py
```

Without renranking or clustering, you can see the raw vector accuracy of the model is clearly better than the current state of the art, which seems to be using the same model just without the MSMT17 dataset!

However, the pixel wise similairty metric alone is insufficient to get the best results.

We need to consider their are other dimensions to the problem that are not being considered, and by finding these metrics, we can improve the accuracy of the model by measuring more than just the vectors in a brute force manner.






