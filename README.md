# VQA_Keras

## Install the following libraries

- tensorflow
- opencv

## Usage

- `python train.py data_dir`
- optionally add `--big-model` for a bigger model

## Code Locations

- `train.py` trains a new model
- `model.py` contains our models
- `prepare_data.py` loads the dataset
- `dataset_gen.py` generates a shape counting dataset. choose parameters at the top of the file.
- `analyze.py` not yet used
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
└───models
    │   model_1.pth
```
