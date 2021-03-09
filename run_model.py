#external library imports
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import os

# project imports
from model import ModelSimple
from load_dataset import CustomDataset



#UTILS METHODS
#------------------------------------------------------------------------------------------------------
@torch.no_grad() #Omit gradient tracking
def get_num_correct(preds, labels_1hot_batch):
    """
    Count the number of hits.
    :preds - softmax predictions, dim -> [batch_size x classes_number].
    :labels_1hot_batch - labels encoded as 1 hot.
    :returns int of number of hits.
    """
    #get max from the softmax values
    max_indices = preds.argmax(dim=1) #dim: [batch_size]

    #to use in index_put_ -> get a tuple of two tensors. One with the rows I want to use (in this case all rows), 
    # the other with column positions which I got from the previous line with argmax.
    indices_row = (torch.tensor(range(max_indices.numel())), max_indices)
    preds_1hot = torch.zeros(preds.shape) #tensor of results
    preds_1hot.index_put_(indices_row, torch.ones(max_indices.numel())) #set 1's in the correct place
    
    return int(torch.sum((labels_1hot_batch == preds_1hot).all(dim=1)))
#------------------------------------------------------------------------------------------------------



#TRAINING METHODS
#------------------------------------------------------------------------------------------------------
def run_training_v1(path_dataset, path_model):
    """
    Example on how to train a model
    """
    root = "./data/" + path_dataset
    epochs = 20
    batch_size = 100

    #DATASET
    transformations = transforms.Compose([transforms.ToTensor()])
    train_set = CustomDataset(root + '/train', transformations)
    train_set_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=batch_size,
                                                    shuffle=False)

    #MODEL                                 
    print(train_set.vocabulary(), train_set.unique_answers)
    model = ModelSimple(len(train_set.vocabulary()), len(train_set.unique_answers)) 

    #OPTIMIZER
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #training loop
    for epoch in range(epochs):

        total_loss = 0
        total_correct = 0

        for batch in train_set_loader:
            # 1. GET BATCH
            #call method __getitem__ from CustomDataset
            images, names, questions, answers, questions_bow, answers_1hot = batch

            # 2. CALCULATE THE LOSS
            preds = model(images, questions_bow) #get predictions
            #convert from 1hot labels into scalar labels (required for the cross entropy function)
            proper_labels = torch.tensor(list(map(lambda x: train_set.to_text(x, False), answers_1hot)))
            loss = F.cross_entropy(preds, proper_labels)

            # 3. UPDATE WEIGHTS
            optimizer.zero_grad() #set the calculation of gradients to zero
            loss.backward() #calculate Gradients
            optimizer.step() #update Weights
        
            # 4. ACCUMULATE LOSS (CONTROL PURPOSES)
            total_loss += loss.item()
            total_correct += get_num_correct(preds, answers_1hot)

        print(
            "epoch:", epoch, 
            "total_correct:", total_correct, 
            "loss:", total_loss
        )
        print(f"Accuracy: {total_correct / len(train_set)}")

    #Save the model
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(model.state_dict(), "./models/" + path_model + ".pth")
#------------------------------------------------------------------------------------------------------



#TESTING METHODS
#------------------------------------------------------------------------------------------------------
@torch.no_grad() #Omit gradient tracking.
def get_all_preds(model, loader):
    """
    Runs the predictions over the entire dataset.
    :param model - model for running the predictions.
    :param loader - loader to retrieve the data.
    :return a tensor with all predictions.
    """
    all_preds = torch.tensor([])
    for batch in loader:
        images, names, questions, answers, questions_bow, answers_1hot = batch #get batch
        preds = model(images, questions_bow) #get predictions
        
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

def testing_v1(path_dataset, path_model):
    """
    Example on how to test a model
    """
    #When running predictions -> disable gradient calculation
    with torch.no_grad():

        #load dataset
        root = "./data/" + path_dataset
        transformations = transforms.Compose([transforms.ToTensor()])
        train_set = CustomDataset(root + '/train', transformations) #load train or val set
        prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=1)

        #create a model
        model = ModelSimple(len(train_set.vocabulary()), len(train_set.unique_answers)) 

        #load weights
        model.load_state_dict(torch.load("./models/" + path_model + ".pth"))
        model.eval()

        #get predictions
        train_preds = get_all_preds(model, prediction_loader) 

        #get ground truth from the dataset
        answers_1hot = list(map(lambda x: train_set.to_categorical(x), train_set.answers_list)) #dim: [questions_total_number]
        answers_1hot = torch.stack(answers_1hot) #to tensor

        #get number of hits in overall
        hits = get_num_correct(train_preds, answers_1hot)

    print('Total correct:', hits)
    print('Accuracy:', hits / len(train_set))
#------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    dataset = "color_single"
    model_name = "color_single"
    print()
    print('training')
    run_training_v1(dataset, model_name)
    
    print()
    print('testing')
    testing_v1(dataset, model_name)
