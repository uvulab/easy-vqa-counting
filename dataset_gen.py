import os
import math
import json
import numpy as np
import cv2
from constants import RED, GREEN, BLUE, CIRCLE, TRIANGLE, RECTANGLE
from image_gen import Image

#EDIT THESE PARAMETERS---------------------------------
dataset_name = "five"
max_shape_count = 5
color_questions = True
shape_questions = True
n_train_images = 8000
n_test_images = 1000
img_size = 64
min_shape_size = 15
allow_overlap = False
balance_factor = math.ceil(max_shape_count / 2.0) #ratio of the most common answer to the least common answer
#should increase along with max_shape_count as the proportion of the least common answer (max) decreases
include_boxes = True #whether to save bounding boxes

noisy_boxes = False #whether to randomly move the bounds to simulate imperfect object detection
noise_bounds = (-5, 15) #min and max boundary adjustment. negative=inward, positive=outward
noise_prob = 0.33 #prob of changing each edge of the bounding box
#------------------------------------------------------

color_names = {RED: "red", GREEN: "green", BLUE: "blue"}
shape_names = {CIRCLE: "circle", TRIANGLE: "triangle", RECTANGLE: "rectangle"}
number_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

def count(shapes, colors, shape, color):
	n = 0
	for (s, c) in zip(shapes, colors):
		if (s == shape or shape == None) and (c == color or color == None):
			n += 1
	return number_names[n]

#Each question is simply a color name, shape name, or both. Answer is a number.
def generate_number_questions(shapes, colors, img_filename):
	result = []
	if max_shape_count > 1:
		result.append(["any", count(shapes, colors, None, None), img_filename])
	if shape_questions:
		for shape in shape_names:
			result.append([shape_names[shape], count(shapes, colors, shape, None), img_filename])
	if color_questions:
		for color in color_names:
			result.append([color_names[color], count(shapes, colors, None, color), img_filename])
	if shape_questions and color_questions:
		for shape in shape_names:
			for color in color_names:
				result.append([color_names[color]+" "+shape_names[shape], count(shapes, colors, shape, color), img_filename])
	return result

def add_noise(x0, y0, x1, y1):
	shifts = np.random.randint(noise_bounds[0], noise_bounds[1], size=4)
	if np.random.random() < noise_prob:
		x0 = max(x0 - shifts[0], 0)
	if np.random.random() < noise_prob:
		y0 = max(y0 - shifts[1], 0)
	if np.random.random() < noise_prob:
		x1 = min(x1 + shifts[2], img_size - 1)
	if np.random.random() < noise_prob:
		y1 = min(y1 + shifts[3], img_size - 1)
	return x0, y0, x1, y1

def generate_data(n):
	questions = []
	images = []
	image_names = []
	annotations = [] #for each image, a list of (x0, y0, x1, y1, class index) for each object
	for i in range(n):
		name = str(i)+".png"
		num_shapes = np.random.randint(max_shape_count)+1
		img = Image(img_size, img_size, num_shapes, min_shape_size, allow_overlap)
		img.generate()
		questions += generate_number_questions(img.shapes, img.colors, name)
		images.append(img.img)
		image_names.append(name)
		if include_boxes:
			a = []
			for i, (x0, y0, x1, y1) in enumerate(img.boxes):
				#for now, class = shape number, see constants.py
				if noisy_boxes:
					x0, y0, x1, y1 = add_noise(x0, y0, x1, y1)
				a.append((x0, y0, x1, y1, img.shapes[i]))
			annotations.append(a)
	return questions, images, image_names, annotations

def balance_answers(questions):
	answer_dict = dict()
	for q in questions:
		if not q[1] in answer_dict:
			answer_dict[q[1]] = []
		answer_dict[q[1]].append(q)
	min_n = min([len(answer_dict[a]) for a in answer_dict])
	max_n = balance_factor * min_n
	result = []
	for a in answer_dict:
		qs = answer_dict[a]
		if len(qs) > max_n:
			chosen_indices = np.random.choice(len(qs), max_n, replace=False)
			for i in chosen_indices:
				result.append(qs[i])
		else:
			result += qs
	return result

def save(questions, images, image_names, annotations, question_path, image_dir):
	with open(question_path, "w") as outfile:
		json.dump(questions, outfile)
	for i in range(len(images)):
		cv2.imwrite(image_dir + image_names[i], images[i])
	if include_boxes:
		with open(image_dir+"_annotations.txt","w") as outfile:
			for i,a in enumerate(annotations):
				outfile.write(image_names[i] + " ")
				for (x0,y0,x1,y1,c) in a:
					outfile.write(str(x0)+","+str(y0)+","+str(x1)+","+str(y1)+","+str(c)+" ")
				outfile.write("\n")
		with open(image_dir+"_classes.txt","w") as outfile:
			for c in range(len(shape_names)):
				#remember, shape constants must be from 0 to num_shapes-1
				#class = shape for now
				outfile.write(shape_names[c] + "\n")

np.random.seed(2021)

train_questions, train_images, train_names, train_annotations = generate_data(n_train_images)
test_questions, test_images, test_names, test_annotations = generate_data(n_test_images)
print("before balancing:")
print(len(train_questions),"train questions,",len(test_questions),"test questions")
train_questions = balance_answers(train_questions)
test_questions = balance_answers(test_questions)
print("after balancing:")
print(len(train_questions),"train questions,",len(test_questions),"test questions")

dataset_dir = "data/"+dataset_name
question_dir = dataset_dir + "/questions"
train_dir = dataset_dir + "/train"
test_dir = dataset_dir + "/test"
os.mkdir(dataset_dir)
os.mkdir(question_dir)
os.mkdir(train_dir)
os.mkdir(test_dir)

save(train_questions, train_images, train_names, train_annotations, question_dir+"/train_questions.json", train_dir+"/")
save(test_questions, test_images, test_names, test_annotations, question_dir+"/test_questions.json", test_dir+"/")

with open(dataset_dir+"/answers.txt", "w") as outfile:
	for i in range(max_shape_count + 1):
		outfile.write(number_names[i] + "\n")
