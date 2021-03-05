import os
import json
import numpy as np
import cv2
from constants import RED, GREEN, BLUE, CIRCLE, TRIANGLE, RECTANGLE
from image_gen import Image

dataset_name = "simple_counting"
n_train_images = 200
n_test_images = 100
max_count = 3
img_size = 64
allow_overlap = False

color_names = {RED: "red", GREEN: "green", BLUE: "blue"}
shape_names = {CIRCLE: "circle", TRIANGLE: "triangle", RECTANGLE: "rectangle"}
number_names = ["zero", "one", "two", "three", "four", "five"]

def count(shapes, colors, shape, color):
	n = 0
	for (s, c) in zip(shapes, colors):
		if (s == shape or shape == None) and (c == color or color == None):
			n += 1
	return number_names[n]

def generate_number_questions(shapes, colors, img_filename):
	result = []
	result.append(["How many shapes?", count(shapes, colors, None, None), img_filename])
	for shape in shape_names:
		result.append(["How many "+shape_names[shape]+"s?", count(shapes, colors, shape, None), img_filename])
	for color in color_names:
		result.append(["How many "+color_names[color]+" shapes?", count(shapes, colors, None, color), img_filename])
	for shape in shape_names:
		for color in color_names:
			result.append(["How many "+color_names[color]+" "+shape_names[shape]+"s?", count(shapes, colors, shape, color), img_filename])
	return result

def generate_data(n):
	questions = []
	images = []
	image_names = []
	for i in range(n):
		name = str(i)+".png"
		num_shapes = np.random.randint(max_count)+1
		img = Image(img_size, img_size, num_shapes, allow_overlap)
		img.generate()
		questions += generate_number_questions(img.shapes, img.colors, name)
		images.append(img.img)
		image_names.append(name)
	return questions, images, image_names

def save(questions, images, image_names, question_path, image_dir):
	with open(question_path, "w") as outfile:
		json.dump(questions, outfile)
	for i in range(len(images)):
		cv2.imwrite(image_dir + image_names[i], images[i])

np.random.seed(2021)

train_questions, train_images, train_names = generate_data(n_train_images)
test_questions, test_images, test_names = generate_data(n_test_images)

dataset_dir = "data/"+dataset_name
question_dir = dataset_dir + "/questions"
train_dir = dataset_dir + "/train"
test_dir = dataset_dir + "/test"
os.mkdir(dataset_dir)
os.mkdir(question_dir)
os.mkdir(train_dir)
os.mkdir(test_dir)

save(train_questions, train_images, train_names, question_dir+"/train_questions.json", train_dir+"/")
save(test_questions, test_images, test_names, question_dir+"/test_questions.json", test_dir+"/")

with open(dataset_dir+"/answers.txt", "w") as outfile:
	outfile.writelines(number_names[0:max_count+1])
