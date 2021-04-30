# Easy VQA Counting with Keras

## Introduction

This project is motivated by the tendency for visual question answering models to struggle with counting questions. [This blog post](https://blog.einstein.ai/interpretable-counting-for-visual-question-answering/) introduces the problem and proposes several specialized counting modules as a solution. Another goal is to provide a small, simple dataset for quick experimentation. Our code is based on [easy-VQA-keras](https://github.com/vzhou842/easy-VQA-keras), but adapted for counting.

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

## Generating Data

Our dataset consists of small images with varying numbers of shapes, which can be 3 kinds of shape (circle, triangle, square) and 3 colors (red, green, blue). Questions are simply the names of the category to count, for example "triangle" for "how many triangles?", "red square" for "how many red squares?", or "any" for "how many shapes total?". Answers are a number.

Let's head to `dataset_gen.py` and walk through generating your first dataset. First, you'll see a series of parameters to edit.
- `dataset_name = "five"`: choose a name for your dataset. A folder data/<dataset_name> will be created for your data.
- `max_shape_count = 5`: the maximum number of shapes to appear per image in this dataset. Each image will have a random number of shapes from 1 to max_shape_count.
- `color_questions = True, shape_questions = True`: whether color and/or shape questions will be included in the dataset. For a simpler task, set only one of these to true. For example, if only color_questions is true, all questions will be "red", "green", or "blue" to count the number of shapes with that color, regardless of shape.
- `n_train_images = 8000, n_test_images = 1000`: the number of train and test images. Note that the dataset will contain multiple questions per image. There should be at least a few thousand train images to ensure sufficient generalization.
- `img_size = 64`: the size in pixels of the square images. For 5 or fewer shapes, 64 should be enough. With more, the image will start to get crowded (an image will only add shapes until no more can fit) and size should be increased.
- `min_shape_size = 15`: the minimum size in pixels of the bounding box of each shape in the image.
- `allow_overlap = False`: whether shapes are allowed to overlap in the image. Setting this to True would make the VQA task more difficult.
- `balance_factor = math.ceil(max_shape_count / 2.0)` the ratio of the most common answer to the least common answer. Initally, every possible question is generated, which will create many more questions with low numbers than high numbers, and incentivize the model to answer a low number all the time. Balance_factor sets a maximum ratio of one answer to another and discards excess questions, so, for example, with max_shape_count = 5, there will be 3 "zero" questions for every "five" question.
- `include_boxes = True`: whether the bounding boxes of the form (x0, y0, x1, y1, shape class) for each shape are saved or not. Should usually be set to True.
- `noisy_boxes = False`: if True, adds noise to the bounding boxes by randomly adjusting the edges. This can simulate VQA with an imperfect object detector.

Once you've chosen your parameters, simply run `python dataset_gen.py` to generate the data.

## Preprocessing the Data

`prepare_data.py` loads the data you just generated into a form the neural network can understand. The setup function returns everything you will need:

- `train_X_ims`: an image for each question. An array of shape (num_questions * img_size * img_size * 3). Num_questions = the size of the dataset.
- `train_X_seqs`: the questions in bag-of-words format. For example, if the vocabulary is `["circle", "triangle", "rectangle", "red", "green", "blue", "any"]`, then the question "red circle" would be `[1, 0, 0, 1, 0, 0, 0]`. Total shape: (num_questions * vocab_size). Note that bag-of-words only works for this dataset because the order of words in the question doesn't matter.
- `train_box_features`: for each question, an array of small images representing the contents of the bounding box for each object in the image, resized to a standard size. Total shape: (num_questions * MAX_COUNT * BOX_SIZE * BOX_SIZE * 3). `MAX_COUNT` is the maximum number of shapes per image that your model can handle. If the image contains fewer shapes, excess boxes will be zeros. `BOX_SIZE` is the (square) size to which all bounding box contents are resized for input to the model. Both of these values are set in `constants.py`.
- `train_box_classes`: if you want a simpler model that only cares about the label of each bounding box, not the constants, use this. In this project, class = shape and the classes are onehot encoded, so for example, `[1, 0, 0]` is "circle". For color information, you will need to look at the actual image. Total shape: (num_questions * MAX_COUNT * 3)
- `train_Y`: the onehot encoded answers, of shape (num_questions * num_answers)
- (Repeat these vectors for the test data)
- `vocab_size`: the number of possible words in the question
- `num_answers`: the number of possible answers (will be from 0 to max_shape_count)
- Any other values returned are not currently used.

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
