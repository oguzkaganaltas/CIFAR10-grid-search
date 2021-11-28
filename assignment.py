# Import the required modules
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

# Fix the randomness
seed = 2385128
torch.manual_seed(seed)


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

HP = {
    "lr": [1e-2,1e-3,1e-5],
    "neurons": [1,2,3],
    "hidden_layers": [[200, 100, 50], [200, 100], [200]],
    "activation_funcs": [nn.ReLU(),nn.Sigmoid()],
    "batch_size":128,
    "num_epoch":100,
    "patience":10
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the ANN
class MyModel(nn.Module):

    def __init__(self, output_shape, hidden_layers,activation):
        super().__init__()
        
        self.activation = activation

        layers = [32*32*1] + hidden_layers + [output_shape]
        self.mods = nn.ModuleList()

        for i in range(len(layers)-1):
            self.mods.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i in range(len(self.mods)-1):
          x = self.activation(self.mods[i](x))
        x = self.mods[-1](x)
        return x

def fit(config):

    patience = HP["patience"]
    
    epoch_took = 0
    best_train_accuracy = 0
    best_train_loss = 0
    best_val_acc = 0
    best_val_loss = 0
    
    for epoch in range(HP["num_epoch"]):

        # Training
        model.train()
        accum_train_loss = 0
        train_correct = train_total = 0
        for n, (imgs, labels) in tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch: {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = loss_function(output, labels)

            # accumlate the loss
            accum_train_loss += loss.item()

            #train accuracy
            train_correct += (torch.argmax(output, dim=1) == labels).sum()
            train_total += labels.size(0) 

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        accum_val_loss = 0
        val_correct = val_total = 0
        with torch.no_grad():
            for x, (imgs, labels) in enumerate(val_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                
                #accumlate the loss
                accum_val_loss += loss_function(output, labels).item()
                
                #validation accuracy
                val_correct += (torch.argmax(output, dim=1) == labels).sum()
                val_total += labels.size(0) 

        val_accuracy = 100 * val_correct / val_total
        train_accuracy = 100 * train_correct / train_total

        train_loss = accum_train_loss / n
        val_loss = accum_val_loss / x

        if(patience == 0):
            break
        elif(val_accuracy <= best_val_acc):
            patience = patience - 1
        else:
            epoch_took = epoch
            best_val_acc = val_accuracy
            best_train_accuracy = train_accuracy
            best_train_loss = train_loss
            best_val_loss = val_loss

            save_dir = f"./best_models/"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/model-{config}.pth")
            patience = HP["patience"]

        print(f'Train Loss = {train_loss:.4f}\tTrain Accuracy = {train_accuracy:.4f}\tVal Loss = {val_loss:.4f}\tVal Accuracy: {val_accuracy:.4f}\tPatience: {patience}')

    return best_train_loss, best_train_accuracy, best_val_loss, best_val_acc, epoch_took

def test(_model):
    # Compute Test Accuracy
    accum_test_loss = 0
    _model.eval()
    with torch.no_grad():
        correct = total = x = 0
        for images, labels in tqdm(test_loader,total=len(test_loader), desc=f"Test: "):
            images, labels = images.to(device), labels.to(device)
            output = _model(images)
            
            accum_test_loss += loss_function(output, labels).item()

            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0)
            x+=1

        test_loss = accum_test_loss / x
        test_accuracy = 100 * correct/total
    print(f'Test Loss= {test_loss:.4f}\tTest Accuracy = {test_accuracy:.4f}%')
    return test_loss, test_accuracy


if __name__ == "__main__":

    train_transform = T.Compose ([
    T.ToTensor(),
    T.Grayscale(),
    T.Normalize(mean =(0.5,), std=(0.5,))
    ])
    val_transform = test_transform = T.Compose([
    T.ToTensor(),
    T.Grayscale(),
    T.Normalize(mean =(0.5,) , std=(0.5,))
    ])

    train_set = CIFAR10 ( root ="CIFAR10", train =True ,transform = train_transform , download = True )
    test_set = CIFAR10 ( root ="CIFAR10", train =False ,transform = test_transform , download = True )
    train_set_length = int(0.8 * len(train_set))
    val_set_length = len(train_set) - train_set_length
    train_set, val_set = random_split(train_set, [train_set_length, val_set_length])


    train_loader = DataLoader(train_set, batch_size=HP["batch_size"], shuffle=True,num_workers=8)
    test_loader = DataLoader(test_set, batch_size=HP["batch_size"],num_workers=8)
    val_loader = DataLoader(val_set, batch_size=HP["batch_size"],num_workers=8)

    results = []
    
    c= 0
    for m in range(len(HP["neurons"])):
        for i in range(len(HP["hidden_layers"])):
            for j in range(len(HP["lr"])):
                for k in range(len(HP["activation_funcs"])):
                    print(f"----- Model: {c} -----")

                    model = MyModel(10, hidden_layers=((np.array(HP["hidden_layers"][i])*HP["neurons"][m]).tolist()),activation=HP["activation_funcs"][k]).to(device)
                    loss_function = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=HP["lr"][j])

                    train_loss, train_accuracy, val_loss, val_accuracy, num_epoch = fit(c) 
                    
                    model_result = [HP["neurons"][m],HP["hidden_layers"][i],HP["lr"][j],HP["activation_funcs"][k], train_loss, train_accuracy, val_loss, val_accuracy, num_epoch+1]
                    PATH = f"./best_models/model-{c}.pth"
                    model.load_state_dict(torch.load(PATH))
                    model_result.append(test(model))
                    print(model_result)
                    results.append(model_result)
                    c=c+1
                
    print(results)
    fields= ["neurons","hidden_layers","lr","activation_funcs","train_loss","train_accuracy","val_loss","val_accuracy","num_epoch","test_loss","test_accuracy"]
    import csv
    with open('RESULTS', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(results)