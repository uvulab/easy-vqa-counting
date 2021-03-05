# VQA_Pytorch

## Install the following libraries

- conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
- conda install opencv
- pip install spacy
  - After the instalation of spacy, run: python -m spacy download en
- pip install torchtext==0.2.3
- pip install -U scikit-learn

## Code Locations

- `run_model.py` runs a toy example
- `model.py` contains our models
- `load_dataset.py` loads the dataset 
- `constants.py`
- `image_gen.py`

## Project 
- See the dataset folder for our format
```
project
│   README.md
│   run_model.py 
|   model.py
|   load_dataset.py
|   constants.py
|   image_gen.py
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
