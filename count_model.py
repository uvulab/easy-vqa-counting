from constants import NUM_SHAPE_CLASSES, MAX_COUNT, BOX_SIZE
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Add
from tensorflow.keras.optimizers import Adam

def build_model_count(im_shape, vocab_size, num_answers, big_model):
	#total inputs: a list of MAX_COUNT (BOX_SIZE * BOX_SIZE * 3) images, each containing the resized
	#contents of one bounding box; followed by the bag-of-words question
	question_input = Input(shape=vocab_size)

	#a repeated block for each image, which given the image and a copy of the question,
	#extracts features from the image, fuses them with the question, and outputs a score
	#of whether the box should be counted for the question
	img_input = Input(shape=(BOX_SIZE, BOX_SIZE, 3))
	question_input_2 = Input(shape=vocab_size)
	img_model = Conv2D(8, 3, padding='same')(img_input)
	img_model = Conv2D(16, 3, padding='same')(img_model)
	img_model = MaxPooling2D()(img_model)
	img_model = Conv2D(32, 3, padding='same')(img_model)
	img_model = MaxPooling2D()(img_model)
	img_model = Flatten()(img_model)
	#fusion happens here
	img_model = Concatenate()([img_model, question_input_2])
	img_model = Dense(32, activation='relu')(img_model)
	img_score = Dense(1, activation='relu')(img_model)
	img_model = Model(inputs=[img_input,question_input_2], outputs=img_score)

	img_inputs = []
	box_scores = []
	for i in range(MAX_COUNT):
		box_input = Input(shape=(BOX_SIZE,BOX_SIZE,3))
		img_inputs.append(box_input)
		score = img_model([box_input, question_input])
		box_scores.append(score)
	#add the scores for each box and convert the total to a onehot encoded output
	score_sum = Add()(box_scores)
	converter = Dense(MAX_COUNT, activation='relu')(score_sum)
	out = Dense(num_answers, activation='softmax')(converter)

	model = Model(inputs=img_inputs + [question_input], outputs=out)
	model.compile(Adam(lr=.0005), loss='categorical_crossentropy', metrics=['accuracy'])

	return model
	
def arrange_inputs_count(images, questions, boxes, box_classes):
	inputs = [boxes[:,i,:,:,:] for i in range(MAX_COUNT)]
	inputs.append(questions)
	return inputs
