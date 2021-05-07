from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import sys
from easy_vqa_model import build_model_easy_vqa, arrange_inputs_easy_vqa
from count_model import build_model_count, arrange_inputs_count
from prepare_data import setup

"""
To use a different model: create a new model file and import new versions of build_model and arrange_inputs
-build_model(im_shape, vocab_size, num_answers, args) creates the model
-arrange_inputs(images, questions, boxes, box_classes) returns a list of desired inputs in order of input to the model
see easy_vqa_model.py for an example
-questions are bag-of-words encoded
-boxes are, for each question, a MAX_COUNT * BOX_SIZE * BOX_SIZE * 3 array representing a list of image slices
at each bounding box, all resized to BOX_SIZE, or zeros if no object is present
-box_classes are, for each question, a MAX_COUNT * NUM_SHAPE_CLASSES (from constants.py) array where each row represents
one box in the image, containing either the one hot encoding of that box's shape class, or zeros if not enough boxes
"""

model_functions = {
	"easy_vqa": (build_model_easy_vqa, arrange_inputs_easy_vqa),
	"count": (build_model_count, arrange_inputs_count)
}

#easy_vqa args: either empty or "--big-model"
#count args: fusion method: "concat", "mul_n", "add_n", or empty for default gated tanh. #optionally add an integer for fusion layer size.

use_boxes = True

if len(sys.argv) < 3:
	print("usage: python train.py data_dir model_name (...args)")
	exit()

data_dir = sys.argv[1]
model_name = sys.argv[2]
if model_name in model_functions:
	build_model, arrange_inputs = model_functions[model_name]
else:
	print("available models:")
	for name in model_functions:
		print(name)
	exit()
args = []
if len(sys.argv) > 3:
	args = sys.argv[3:]

# Prepare data
train_X_ims, train_X_seqs, train_box_features, train_box_classes, train_Y, train_image_ids, \
test_X_ims, test_X_seqs, test_box_features, test_box_classes, test_Y, test_image_ids, \
im_shape, vocab_size, num_answers, _, _, _ = setup(data_dir, use_boxes=use_boxes)

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, args)
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
