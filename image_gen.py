import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from constants import MIN_SIZE, RED, GREEN, BLUE, CIRCLE, TRIANGLE, RECTANGLE

colors = dict()
colors[RED] = (255, 0, 0)
colors[GREEN] = (0, 255, 0)
colors[BLUE] = (0, 0, 255)

class Image:
	def __init__(self, w, h, max_num_shapes, allow_overlap):
		self.w = w
		self.h = h
		self.max_num_shapes = max_num_shapes
		self.allow_overlap = allow_overlap
		self.img = np.zeros((h, w, 3),dtype=np.uint8)
		self.free_spaces = []
		for x in range(self.w):
			for y in range(self.h):
				w = self.w - x
				h = self.h - y
				if w >= MIN_SIZE and h >= MIN_SIZE:
					self.free_spaces.append((x, y, w, h))
		#self.generate()

	def generate(self):
		self.shapes = []
		self.colors = []
		self.boxes = []
		while len(self.shapes) < self.max_num_shapes and len(self.free_spaces) > 0:
			shape = np.random.choice([CIRCLE, TRIANGLE, RECTANGLE])
			color = np.random.choice([RED, GREEN, BLUE])
			box = self.place_shape(shape, color)
			self.shapes.append(shape)
			self.colors.append(color)
			self.boxes.append(box)

	#randomly places the shape in free space. returns the bounding box in YOLO format, or None if there is no space.
	def place_shape(self, shape, color):
		if len(self.free_spaces) == 0:
			return None
		(x, y, fw, fh) = self.free_spaces[np.random.randint(len(self.free_spaces))]
		if shape == CIRCLE or shape == TRIANGLE: #symmetric and odd
			w = np.random.randint(MIN_SIZE, min(fw,fh)+1)
			if w % 2 == 0:
				w -= 1
			h = w
		else:
			w = np.random.randint(MIN_SIZE, fw+1)
			h = np.random.randint(MIN_SIZE, fh+1)
		if shape == CIRCLE:
			self.draw_circle(color, x + w // 2, y + w // 2, w // 2)
		if shape == TRIANGLE:
			self.draw_triangle(color, x + w // 2, y + w // 2, w // 2, 2 * math.pi * np.random.rand())
		if shape == RECTANGLE:
			self.draw_rectangle(color, x, y, w, h)
		if not self.allow_overlap:
			self.update_free_space(x, y, w, h)
		return (x, y, x + w - 1, y + h - 1)

	#if the given bounding box is occupied, updates the remaining maximum available bounding boxes
	def update_free_space(self, bx, by, bw, bh):
		new_free_spaces = []
		for (x, y, w, h) in self.free_spaces:
			new_w = w
			new_h = h
			#coordinate is inside the bounding box
			if x >= bx and x < bx + bw and y >= by and y < by + bh:
				pass
			else:
				#bounding box is to the right, restrict free width
				if x < bx and max(y, by) < min(y + h, by + bh):
					new_w = min(w, bx - x)
				#bounding box is above, restrict free height
				if y < by and max(x, bx) < min(x + w, bx + bw):
					new_h = min(h, by - y)
				if new_w >= MIN_SIZE and new_h >= MIN_SIZE:
					new_free_spaces.append((x, y, new_w, new_h))
		self.free_spaces = new_free_spaces

	def draw_circle(self, color, cx, cy, r):
		cv2.circle(self.img, (cx, cy), r, colors[color], -1)

	#cx, cy, r specify a circle surrounding the triangle. Theta specifies the rotation in radians, with 0 being upright.
	def draw_triangle(self, color, cx, cy, r, theta=0):
		p0 = [cx + r*math.cos(math.pi/2 + theta), cy + r*math.sin(math.pi/2 + theta)]
		p1 = [cx + r*math.cos(7*math.pi/6 + theta), cy + r*math.sin(7*math.pi/6 + theta)]
		p2 = [cx + r*math.cos(-math.pi/6 + theta), cy + r*math.sin(-math.pi/6 + theta)]
		cv2.fillPoly(self.img, np.int32(np.array([[p0, p1, p2]])), colors[color])

	def draw_rectangle(self, color, x, y, w, h):
		cv2.rectangle(self.img, (x,y), (x+w-1, y+h-1), colors[color], -1)

	def show(self):
		plt.imshow(self.img)
		plt.show()

if __name__ == "__main__":
	"""
	img = Image(40, 40)
	img.draw_circle(RED, 10, 10, 10)
	img.draw_triangle(GREEN, 30, 20, 10, theta=math.pi)
	img.draw_rectangle(BLUE, 5, 25, 15, 5)
	img.show()
	"""
	for _ in range(5):
		img = Image(64, 64, 5, False)
		img.generate()
		img.show()

