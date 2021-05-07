from constants import NUM_SHAPE_CLASSES, MAX_COUNT, BOX_SIZE
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, Add, Multiply
from tensorflow.keras.optimizers import Adam

#args: fusion method, either "concat", "mul_288", "mul_n", "add_288", "add_n", or defaults to gated tanh if empty
def build_model_count(im_shape, vocab_size, num_answers, args):
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

	#choose a fusion method
	if len(args) > 0 and args[0] == "concat":
		img_model = Concatenate()([img_model, question_input_2])
		img_model = Dense(32, activation='relu')(img_model)
		img_score = Dense(1, activation='sigmoid')(img_model)

	elif len(args) > 0 and args[0] == "mul_288":
		#note: this only works if the flattened convolutional output has size 288, image size 64
		question = Dense(288, activation='relu')(question_input_2)
		img_model = Multiply()([question, img_model])
		img_model = Dense(32, activation='relu')(img_model)
		img_score = Dense(1, activation='sigmoid')(img_model)
		
	elif len(args) > 0 and args[0] == "mul_n":
		dense_size = int(args[1])

		img_model = Dense(dense_size, activation='tanh')(img_model)
		question = Dense(dense_size, activation='tanh')(question_input_2)
		img_model = Multiply()([question, img_model])
		img_model = Dense(32, activation='relu')(img_model)
		img_score = Dense(1, activation='sigmoid')(img_model)
		
	elif len(args) > 0 and args[0] == "add_n":
		dense_size = int(args[1])

		img_model = Dense(dense_size, activation='tanh')(img_model)
		question = Dense(dense_size, activation='tanh')(question_input_2)
		img_model = Add()([question, img_model])
		img_model = Dense(32, activation='relu')(img_model)
		img_score = Dense(1, activation='sigmoid')(img_model)
		
	elif len(args) > 0 and args[0] == "add_288":
		question = Dense(288, activation='relu')(question_input_2)
		img_model = Add()([question, img_model])
		img_model = Dense(32, activation='relu')(img_model)
		img_score = Dense(1, activation='sigmoid')(img_model)
	#default: gated tanh	
	else:
		dense_size = int(args[0])

		#(1) Gated tanh of the image
		y_hat_img = Dense(dense_size, activation='tanh')(img_model)
		g_img = Dense(dense_size, activation='sigmoid')(img_model)
		img_model = Multiply()([y_hat_img, g_img])

		#(2) Gated tanh of the question
		y_hat_question = Dense(dense_size, activation='tanh')(question_input_2)
		g_question = Dense(dense_size, activation='sigmoid')(question_input_2)
		y_question = Multiply()([y_hat_question, g_question])

		#fusion happens here
		img_model = Multiply()([y_question, img_model])

		y_hat_dense1 = Dense(32, activation='tanh')(img_model)
		g_dense1 = Dense(32, activation='sigmoid')(img_model)
		img_model = Multiply()([y_hat_dense1, g_dense1])

		y_hat_dense2 = Dense(1, activation='tanh')(img_model)
		g_dense2 = Dense(1, activation='sigmoid')(img_model)
		img_score = Multiply()([y_hat_dense1, g_dense1])
	
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
