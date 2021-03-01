import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.feature_extraction.text import CountVectorizer
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np
import spacy
import string


class CustomDataset(Dataset):
    def __init__(self, folder_path : str, transformations : transforms):
        """
        Read image file names.
        :param folder_path - path to image folder.
        :param transformations - transformation to be applied to the images.
        :return - None.
        """
        # Save the transformations
        self.transformations = transformations


        # LOAD QUESTIONS
        #-----------
        #read question from the corresponding folder
        index_root = folder_path.rfind('/')
        set_name = folder_path[index_root+1:]
        questions_path = folder_path[:index_root] + f'/questions/{set_name}_questions.json'

        with open(questions_path, 'r') as file:
            qs = json.load(file)
        self.questions_list = [q[0] for q in qs]
        self.answers_list = [q[1] for q in qs]
        self.img_names_list = [q[2] for q in qs]

        #load the total number of unique answers
        with open(folder_path[:index_root] + '/answers.txt', 'r') as file:
            all_answers = [a.strip() for a in file]
        self.unique_answers = all_answers
        #-----------


        # BAG OF WORDS (bow)
        #-----------
        self.nlp = spacy.load("en_core_web_sm")
        self.punctuations = string.punctuation #punctuation marks
        #variable to transform into bow
        self.transformer = CountVectorizer(tokenizer = self.spacy_tokenizer, ngram_range=(1,1))

        tokenized_questions = []
        for question in self.questions_list:
            doc_tokens = self.spacy_tokenizer(question)
            tokenized_questions.append(doc_tokens)
        #create a single/flatten list from the list of list 
        flatten_tokens = [item for sublist in tokenized_questions for item in sublist]
        self.transformer.fit(flatten_tokens)
        #-----------


        #LOAD IMAGES
        #-----------
        # Get image list (exclude files starting with '_')
        file_names = glob.glob(folder_path + '/[!_]*')
        self.image_list = []
        for name in self.img_names_list:
            self.image_list.append(folder_path + '/' + name)

        # Calculate len
        self.data_len = len(self.image_list)

        #image shape according to torch.shape, i.e., channels x height x width
        shape = np.asarray(Image.open(self.image_list[0])).shape
        self.im_shape = (shape[2],shape[0],shape[1])
        #-----------


    def __getitem__(self, index : int):
        """
        Read image at index and find the question, answer, and image related to that index.
        :param index - index of the item I want to get
        :return image - torch tensor of the image.
        :return img_name - image name.
        :return question_text - question in plain text.
        :return answer_text - answer in plain text.
        :return questions_bow - torch tensor of the question.
        :return answer_1hot - torch tensor of the answer.
        """
        #GET IMAGE
        #-----------
        single_image_path = self.image_list[index] #get full path + filename
        img_name = self.img_names_list[index] #get just filename 
        # Open image
        image = Image.open(single_image_path)
        image = self.transformations(image)  
        #-----------

        #GET QUESTIONS AND ANSWERS - PLAIN TEXT AND ENCODED
        #-----------
        #questions
        question_text = self.questions_list[index]

        #get token from spacy and then transform into bag of words encoding
        tokens = self.spacy_tokenizer(question_text)
        questions_bow = self.transformer.transform([' '.join(tokens)])
        questions_bow = torch.tensor(questions_bow.toarray(), dtype=torch.float32) #dim = [1,x]
        questions_bow = questions_bow.squeeze() #dim = [x]

        #answers
        answer_text = self.answers_list[index]
        answer_1hot = self.to_categorical(answer_text)
        answer_1hot = torch.tensor(answer_1hot)
        #-----------

        return (image, img_name, question_text, answer_text, questions_bow, answer_1hot)
    

    def to_categorical(self, answer_text : string):
        """ 
        1-hot encodes a tensor / similar to keras.utils.to_categorical.
        :param answer_text - from the categories/answers, my category/answer.
        :return 1-hot category.
        """
        #important -> to use unique_answers and not answers_list
        answer_index = self.unique_answers.index(answer_text)
        number_of_classes = len(self.unique_answers)
        return torch.eye(number_of_classes, dtype=torch.float32)[answer_index]


    def to_text(self, answer_1hot, get_text):
        """
        Transform the one hot answer into the text answer or number answer.
        :param answer_1hot - torch tensor of size [1, 1hot_encoding_size]
        :return string of the corresponding word.
        """
        #get all categories
        categories_1hot = torch.eye(len(self.unique_answers)) #dim = [k x k]
        #get if all elements are set to TRUE (.all)
        results = (categories_1hot == answer_1hot).all(dim=1) #dim = [k]
        #from the TRUE/FALSE results get the only '1'
        index = results.argmax() 
        if get_text:
            return self.unique_answers[index]
        else:
            return index


    def spacy_tokenizer(self, question):
        """
        Receive one question and creates tokens using spacy.
        :param question - sentence to be tokenized.
        :return all tokens generated from the question.
        """
        # tokenize the question
        doc_tokens = self.nlp(question)
        # Lowercase and removing punctuations
        doc_tokens = [ word.lower_ for word in doc_tokens if word.lower_ not in self.punctuations ]
        return doc_tokens


    def __len__(self):
        """
        Gets the size of the dataset.
        :return number of elements in this dataset.
        """
        return self.data_len


    def vocabulary(self):
        """
        Get the vocabulary from the bag of words we have.
        :return dictionary of the vocabulary.
        """
        return self.transformer.vocabulary_




''' 
#testing cases
if __name__ == "__main__":
    """
    Example of how to use CustomDataset
    """
    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor()])

    # Define custom dataset
    batch_size = 3
    train_set = CustomDataset('./data/simple_images/train', transformations)


    # Define data loader (to create and access batches)
    train_set_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=batch_size,
                                                    shuffle=False)
    
    # How get a batch of images
    
    for images, img_names, qs_texts, as_texts, qs_encoded, as_1hot in train_set_loader:
        # Feed the data to the model
        

        #questions_bow, answers_onehot = train_set.get_tensors(questions_texts, answers_texts)
        print()
        a = train_set.to_text(as_1hot[0])
        break
    

    
    # How to visualize one image
    tensor_temp = train_set[0][0] #access tuple, then, access the image

    #we need to permute because the tensor is: batch x height x width 
    plt.imshow( tensor_temp.permute(1, 2, 0)  )
    plt.show()
    #
'''

