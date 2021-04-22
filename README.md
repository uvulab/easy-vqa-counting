# Easy VQA Counting Keras

## Introduction

This project is motivated by the tendency for visual question answering models to struggle with counting questions. [This blog post](https://blog.einstein.ai/interpretable-counting-for-visual-question-answering/) introduces the problem and proposes several specialized counting modules as a solution. Another goal is to provide a small, simple dataset for quick experimentation. Our code is based on [easy-VQA-keras](https://github.com/vzhou842/easy-VQA-keras), but adapted for counting.

Our dataset consists of small images with varying numbers of shapes, which can be 3 kinds of shape and 3 colors. Questions are simply the names of the category to count, for example "triangle" for "how many triangles?", "red square" for "how many red squares?", or "any" for "how many shapes total?". Answers are a number.

We currently include two models. The `easy_vqa` model, taken from easy-VQA-keras, is a simple convolutional network over the entire image, and fails to count beyond one. This failure demonstrates the need for specialized counting network. The `count` model, similar to SoftCount in the blog post, takes a list of small images representing each object within its bounding box. Then each object receives a score of how well it fits the question, and the scores are summed to produce a count. This model can successfully learn to count (at least to 10) but still has limitations. It cannot handle questions that depend on an object's relationship to another object, and it requires the ground-truth bounding boxes. Realistically, an object detector may return multiple boxes for the same object, and non-maximum suppression is needed.

We hope this code will provide a starting point to investigate more challenging counting problems.

## Install the following libraries

- tensorflow
- opencv

## Usage

- first, `python dataset_gen.py` to generate data.
- `python train.py data_dir model_name` to train a model.
- optionally add `--big-model` for a bigger model, if using the easy_vqa model
- to run with the default data: `python train.py data/five count`

## Code Locations

- `train.py` trains a new model
- `easy_vqa_model.py` and `count_model.py` contain our models. See the instructions in `train.py` to add your own model.
- `prepare_data.py` loads the dataset
- `dataset_gen.py` generates a shape counting dataset. choose parameters at the top of the file.
- `constants.py` contains color and shape constants
- `image_gen.py` creates colored shape images

## Project 
- See the dataset folder for our format
```
project
│   README.md
│   train.py 
|   etc
│
└───data
│   │
│   └───simple_images
│       │   answers.txt
│       │  
│       └───questions
│       │   test_questions.json
|       |   train_questions.json
│       │  
|       └───train
|       |   _annotations.txt
|       |   _classes.txt
|       |   img1.png
|       |   img2.png
|       |   ...
│       │  
|       └───valid
|       |   _annotations.txt
|       |   _classes.txt
|       |   img7.png
|       |   img8.png
|       |   ...
│   
```
