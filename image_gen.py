import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from constants import RED, GREEN, BLUE

colors = dict()
colors[RED] = (255, 0, 0)
colors[GREEN] = (0, 255, 0)
colors[BLUE] = (0, 0, 255)

class Image:
	def __init__(self, w, h):
		self.w = w
		self.h = h
		self.img = np.zeros((h, w, 3),dtype=np.uint8)

	def draw_circle(self, color, cx, cy, r):
		cv2.circle(self.img, (cx, cy), r, colors[color], -1)

	#cx, cy, r specify a circle surrounding the triangle. Theta specifies the rotation in radians, with 0 being upright.
	def draw_triangle(self, color, cx, cy, r, theta=0):
		p0 = [cx + r*math.cos(math.pi/2 + theta), cy + r*math.sin(math.pi/2 + theta)]
		p1 = [cx + r*math.cos(7*math.pi/6 + theta), cy + r*math.sin(7*math.pi/6 + theta)]
		p2 = [cx + r*math.cos(-math.pi/6 + theta), cy + r*math.sin(-math.pi/6 + theta)]
		cv2.fillPoly(self.img, np.int32(np.array([[p0, p1, p2]])), colors[color])

	def draw_rectangle(self, color, x, y, w, h):
		cv2.rectangle(self.img, (x,y), (x+w, y+h), colors[color], -1)

	def show(self):
		plt.imshow(self.img)
		plt.show()

if __name__ == "__main__":
	img = Image(40, 40)
	img.draw_circle(RED, 10, 10, 10)
	img.draw_triangle(GREEN, 30, 20, 10, theta=math.pi)
	img.draw_rectangle(BLUE, 5, 25, 15, 5)
	img.show()
