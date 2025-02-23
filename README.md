# EveryID: a repository for the recognition of people, objects, scenes and events.

![EveryID](img/EveryID.jpg)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/)
[![transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-pink)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pydantic](https://img.shields.io/badge/pydantic-v2.5-ff3399.svg)](https://docs.pydantic.dev/)
[![Typing](https://img.shields.io/badge/Typing-Supported-brightgreen.svg)](https://docs.python.org/3/library/typing.html)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Pillow](https://img.shields.io/badge/Pillow-8.0%2B-brightgreen.svg)](https://python-pillow.org/)

## Table of Contents
- [Vision](#vision)
- [Applications](#applications)
  - [TV Post Production](#tv-post-production)
  - [Surveillance](#surveillance)
  - [Government Intelligence](#government-intelligence-and-defence)
  - [Agent Based AI](#revolution-in-agent-based-ai-frameworks)
- [Fundamental Challenges](#fundamental-challenges)
- [Proposed Solutions](#how-i-propose-to-solve-these-challenges)
- [Setup](#initial-setup)
- [Person ReID](#person-reid)
  - [Training a Model](#training-a-model)
  - [Uploading a Model](#uploading-a-model-to-the-hub)
  - [Downloading a Model](#download-a-model)
  - [Testing the Model](#test-the-model)

## Vision

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

### TV Post Production: 

The ability to automatically index and search through video content revolutionizes post-production workflows:

- Automatic logging and syncing of timecode across multiple footage sources
- Instant searchability of footage based on people, scenes, and concepts
- Knowledge graph generation for person, scene, and concept clusters
- Context-aware tracking that considers probabilistic likelihood over frame-by-frame matching
- Elimination of manual footage review processes
- Real-time indexing of large productions
- Automated organization of footage by cast members, locations, and scene types

### Surveillance: 

The ability to track and identify individuals across multiple cameras and time periods is crucial for modern surveillance systems. Current solutions often fail when dealing with real-world challenges like varying lighting conditions, different camera angles, partial occlusions, and changes in appearance. This framework provides robust person re-identification capabilities that can:

- Track individuals across non-overlapping camera networks
- Maintain identity consistency despite appearance changes
- Handle crowded scenes and occlusions
- Process real-time video streams efficiently
- Scale to large camera networks without performance degradation

### Government intelligence and defence:

Intelligence agencies face the complex task of analyzing vast amounts of visual data to identify persons of interest across diverse sources and timeframes. This framework offers:

- Rapid processing of surveillance footage and imagery
- Cross-referencing capabilities across multiple data sources
- Temporal analysis of movement patterns
- Integration with existing intelligence systems
- Secure, auditable tracking of identifications
- Ability to handle low-quality or degraded imagery

### Revolution in agent based AI frameworks:

AI agents at present rely on retrieival of information to help them answer questions, organise and display data , or transmit it somewhere and much more. However what multi agent systems can't really do is make actually informed predictions of future events, outside of logical inference from the training data for simple things.

But, for specific planning pipelines that need to be made in rapid succession and optimised in real time, for companies or government agencies, the current retrieval paradigm is lackluster at best.

Intuively to me, we need to move to the recognition paradigm, which places a focus on models and LLM agents which recognise sequences of patterns everywhere in data across time instead of particular patterns period, allowing the generation of the lightning bolt to perseverate beyond its genesis, like the way the bolt finds the cloest piece of metal or tree, through recongition we can build agents that predict near into the futrue based on what something is likely to be as opposed to what has happened or is most similiar in the naieve RAG sense.

This will allow us to move to agents that can string complex plans over time and dynamically change such plans in repsonse to new information , essentially allowing for self reinforcement from faliure, like the Anterior cinculate cortex in the human brain. Their needs to be a real definition of good and bad across time and space , which requires eyes , attention and a true objective function.

Therefore although the tangible goal at present is to solve person REID, the overarching goal is to aid in pushing forward the recognition regime of agents.

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

My conviction is that time and localisation in space with the help of person tracking and general temporal flow of events, although abstract , plays a major role in perhaps averaging the right person.

If the model can get us to high top rank accuracies, and we can track the flow of events in time and space, then I believe
the accuracy of the person reid system will skyrocket, conveying how the model is only half of this age old problem.






