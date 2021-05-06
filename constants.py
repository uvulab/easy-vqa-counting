#shape constants should be from 0 to num_shapes-1 so that they can correspond to classes
CIRCLE = 0
TRIANGLE = 1
RECTANGLE = 2

RED = 3
GREEN = 4
BLUE = 5

NUM_SHAPE_CLASSES = 3
MAX_COUNT = 10 #the maximum number of shapes our network can handle. Should be at least as high as the maximum number of shapes in the dataset
BOX_SIZE = 15 #the standard size of resized bounding boxes for input to the network
