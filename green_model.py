from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Nadam
import numpy as np

def build_model_green(data_size, num_answers):
  # The question network
  data_input = Input(shape=data_size)
  o = Dense(32, activation='tanh')(data_input)
  o = Dense(32, activation='tanh')(o)
  o = Dense(num_answers, activation='softmax')(o)

  model = Model(inputs=[data_input], outputs=o)
  model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])

  return model

def arrange_inputs_green(data):
    #data has dim of [num_questions x MAX_COUNT x NUM_SHAPE_CLASSES]
    # we have to get the sum, return will be [num_questions x NUM_SHAPE_CLASSES]
	return [np.sum(data, axis=1)]
