from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import sys
from easy_vqa_model import build_model_easy_vqa, arrange_inputs_easy_vqa
from count_model import build_model_count, arrange_inputs_count
from prepare_data import setup

"""
To use a different model: create a new model file and import new versions of build_model and arrange_inputs
build_model(im_shape, vocab_size, num_answers, big_model) creates the model
arrange_inputs(images, questions, boxes, box_classes) returns a list of desired inputs in order of input to the model
see easy_vqa_model.py for an example
questions are bag-of-words encoded
boxes will be the box coordinates and/or image slices, not used yet
box_classes are each a MAX_COUNT * NUM_SHAPE_CLASSES (from constants.py) array where each row represents
one box in the image, containing either the one hot encoding of that box's shape class, or zeros if not enough boxes
"""
build_model = build_model_count
arrange_inputs = arrange_inputs_count
use_boxes = True

if len(sys.argv) < 2:
	print("usage: python train.py data_dir (--big-model)")
	exit()

data_dir = sys.argv[1]
big_model = len(sys.argv) >= 3 and sys.argv[2] == "--big-model"


if big_model:
  print('Using big model')

# Prepare data
train_X_ims, train_X_seqs, train_box_features, train_box_classes, train_Y, train_image_ids, \
test_X_ims, test_X_seqs, test_box_features, test_box_classes, test_Y, test_image_ids, \
im_shape, vocab_size, num_answers, _, _, _ = setup(data_dir, use_boxes=use_boxes)

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, big_model)
model.summary()
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

print('\n--- Training model...')
model.fit(
  arrange_inputs(train_X_ims, train_X_seqs, train_box_features, train_box_classes),
  train_Y,
  validation_data=(arrange_inputs(test_X_ims, test_X_seqs, test_box_features, test_box_classes), test_Y),
  shuffle=True,
  epochs=50,
  batch_size=32,
  callbacks=[checkpoint],
)
