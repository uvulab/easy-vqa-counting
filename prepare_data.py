from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import json
import os
import numpy as np
from constants import NUM_SHAPE_CLASSES, MAX_COUNT
#from easy_vqa import get_train_questions, get_test_questions, get_train_image_paths, get_test_image_paths, get_answers

def setup(data_dir, use_boxes=False):
  print('\n--- Reading questions...')
  #if use_data_dir:
  def read_questions(path):
    with open(path, 'r') as file:
      qs = json.load(file)
    texts = [q[0] for q in qs]
    answers = [q[1] for q in qs]
    image_ids = [int(q[2][:-4]) for q in qs]#[q[2] for q in qs]
    return (texts, answers, image_ids)
  train_qs, train_answers, train_image_ids = read_questions(data_dir+'/questions/train_questions.json')
  test_qs, test_answers, test_image_ids = read_questions(data_dir+'/questions/test_questions.json')
  #else:
    # Use the easy-vqa package
    #train_qs, train_answers, train_image_ids = get_train_questions()
    #test_qs, test_answers, test_image_ids = get_test_questions()
  print(f'Read {len(train_qs)} training questions and {len(test_qs)} testing questions.')


  print('\n--- Reading answers...')
  #if use_data_dir:
  with open(data_dir+'/answers.txt', 'r') as file:
    all_answers = [a.strip() for a in file]
  #else:
    # Read answers from the easy-vqa package
    #all_answers = get_answers()
  num_answers = len(all_answers)
  print(f'Found {num_answers} total answers:')
  print(all_answers)


  print('\n--- Reading/processing images...')
  def load_and_proccess_image(image_path):
    # Load image, then scale and shift pixel values to [-0.5, 0.5]
    im = img_to_array(load_img(image_path))
    return im / 255 - 0.5

  def read_images(paths):
    # paths is a dict mapping image ID to image path
    # Returns a dict mapping image ID to the processed image
    ims = {}
    for image_id, image_path in paths.items():
      ims[image_id] = load_and_proccess_image(image_path)
    return ims

  #if use_data_dir:
  def extract_paths(dir):
    paths = {}
    for filename in os.listdir(dir):
      if filename.endswith('.png'):
        image_id = int(filename[:-4])
        paths[image_id] = os.path.join(dir, filename)
    return paths

  #a dict mapping image ID to a list of(x0, y0, x1, y1, class_id) bounding boxes
  def read_boxes(dir):
    result = dict()
    with open(dir + '/_annotations.txt') as file:
      for line in file:
        #be sure the image filenames don't have spaces!
        args = line.strip().split(" ")
        image_id = int(args[0][:-4]) #remove the .png
        boxes = []
        for i in range(1, len(args)):
          a = args[i].split(",")
          box = ()
          for n in a:
            box += (int(n),)
          boxes.append(box)
        result[image_id] = boxes
    return result

  #a sequence of MAX_COUNT vectors. The ith vector contains the onehot-encoded class of the ith box,
  #or zeros if fewer than i boxes
  def format_box_classes(box_list):
    result = np.zeros((MAX_COUNT, NUM_SHAPE_CLASSES))
    for i, box in enumerate(box_list):
      result[i][box[4]] = 1.0
    return result

  train_ims = read_images(extract_paths(data_dir+'/train'))
  test_ims  = read_images(extract_paths(data_dir+'/test'))
  #else:
    # Read images from the easy-vqa package
    #train_ims = read_images(get_train_image_paths())
    #test_ims = read_images(get_test_image_paths())
  im_shape = train_ims[0].shape
  print(f'Read {len(train_ims)} training images and {len(test_ims)} testing images.')
  print(f'Each image has shape {im_shape}.')

  print('\n--- Fitting question tokenizer...')
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(train_qs)

  # We add one because the Keras Tokenizer reserves index 0 and never uses it.
  vocab_size = len(tokenizer.word_index) + 1
  print(f'Vocab Size: {vocab_size}')
  print(tokenizer.word_index)


  print('\n--- Converting questions to bags of words...')
  train_X_seqs = tokenizer.texts_to_matrix(train_qs)
  test_X_seqs = tokenizer.texts_to_matrix(test_qs)
  print(f'Example question bag of words: {train_X_seqs[0]}')


  print('\n--- Creating model input images...')
  train_X_ims = np.array([train_ims[id] for id in train_image_ids])
  test_X_ims = np.array([test_ims[id] for id in test_image_ids])


  print('\n--- Creating model outputs...')
  train_answer_indices = [all_answers.index(a) for a in train_answers]
  test_answer_indices = [all_answers.index(a) for a in test_answers]
  train_Y = to_categorical(train_answer_indices)
  test_Y = to_categorical(test_answer_indices)
  print(f'Example model output: {train_Y[0]}')

  train_box_classes = None
  test_box_classes = None
  if use_boxes:
    train_boxes = read_boxes(data_dir+'/train')
    test_boxes = read_boxes(data_dir+'/test')
    train_box_classes = np.array([format_box_classes(train_boxes[id]) for id in train_image_ids])
    test_box_classes = np.array([format_box_classes(test_boxes[id]) for id in test_image_ids])

  #TODO: include box coordinates
  return (train_X_ims, train_X_seqs, train_box_classes, train_Y, test_X_ims, test_X_seqs,
          test_box_classes, test_Y, im_shape, vocab_size, num_answers,
          all_answers, test_qs, test_answer_indices)  # for the analyze script