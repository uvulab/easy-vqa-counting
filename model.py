#external library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset

# project imports
from load_dataset import CustomDataset



class ModelSimple(nn.Module):

    def __init__(self, vocabulary_size : int, num_answers : int):
        """
        Define all the layers that have weights.
        :param vocabulary_size - size of the vocabulary of our NLP. 
         Since we are using bag of words, one fully connected layer is related to this size.
        :param num_answers - this will indicate the output of our model.
        :return None.
        """
        super().__init__()
        #image model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=(1,1), stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=(1,1), stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=(1,1), stride=1)
        self.fc1_image = nn.Linear(in_features=32*8*8, out_features=32)

        #question model
        self.vocabulary_size = vocabulary_size
        self.fc1_question = nn.Linear(in_features=vocabulary_size, out_features=32)
        self.fc2_question = nn.Linear(in_features=32, out_features=32)

        #merged model
        self.fc1_general = nn.Linear(in_features=32, out_features=32)
        self.fc2_general = nn.Linear(in_features=32, out_features=num_answers)


    def forward(self, input_img, input_question):
        """
        Forward method to be called when training.
        :param input_img - input image. It has to be with batch dim, e.g., [batch x channels x height x width]
        :param input_question - input question (bag of words -bow- ). It has to be with batch dim, e.g., [batch x bow_size]
        :return softmax with size -> len(train_set.unique_answers).
        """
        # IMAGE MODEL
        #---------
        img_model = F.relu(self.conv1(input_img))
        img_model = F.max_pool2d(img_model, kernel_size=2, stride=2)

        img_model = F.relu(self.conv2(img_model))
        img_model = F.max_pool2d(img_model, kernel_size=2, stride=2)

        img_model = F.relu(self.conv3(img_model))
        img_model = F.max_pool2d(img_model, kernel_size=2, stride=2)

        img_model = img_model.reshape(-1, 32*8*8)
        img_model = F.tanh(self.fc1_image(img_model))
        #---------

        # QUESTION MODEL
        #---------
        question_model = F.tanh(self.fc1_question(input_question))
        question_model = F.tanh(self.fc2_question(question_model))
        #---------

        # MERGED MODEL
        #---------
        merged_model = torch.mul(img_model, question_model)
        merged_model = F.tanh(self.fc1_general(merged_model))
        merged_model = F.softmax(self.fc2_general(merged_model))
        #---------

        return merged_model




"""
#testing cases
if __name__ == "__main__":
    #------------------------------------------------------------
    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor()])

    # Define custom dataset
    batch_size = 2
    train_set = CustomDataset('./data/simple_images/train', transformations)


    # Define data loader (to create and access batches)
    train_set_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=batch_size,
                                                    shuffle=False)
    #-------------------------------------------------------------

    torch.set_grad_enabled(False)
    model = ModelSimple(len(train_set.vocabulary()), len(train_set.unique_answers))


    sample = next(iter(train_set))
    image, img_name, qs_text, as_text, qs_bow, as_1hot = sample
    image = image.unsqueeze(0)
    qs_bow = qs_bow.unsqueeze(0)
    pred = model(image, qs_bow)

    print(pred.shape)
#"""