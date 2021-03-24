from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import sys
from model import build_model
from prepare_data import setup

if len(sys.argv) < 2:
	print("usage: python train.py data_dir (--big-model)")
	exit()

data_dir = sys.argv[1]
big_model = len(sys.argv) >= 3 and sys.argv[2] == "--big-model"

# Support command-line options
#parser = argparse.ArgumentParser()
#parser.add_argument('--big-model', action='store_true', help='Use the bigger model with more conv layers')
#parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
#args = parser.parse_args()

if big_model:
  print('Using big model')
#if args.use_data_dir:
#  print('Using data directory')

# Prepare data
train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, _, _, _ = setup(data_dir)

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, big_model)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

print('\n--- Training model...')
model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=40,
  callbacks=[checkpoint],
)
