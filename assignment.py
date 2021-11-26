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
    "num_epoch":30,
    "patience":5
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

def fit(m,i,j,k):
    patience = HP["patience"]
    prev_val_loss = np.inf
    best_model_at_epoch = 0
    for epoch in range(HP["num_epoch"]):
        # Training
        model.train()
        accum_train_loss = 0
        for n, (imgs, labels) in tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch: {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = loss_function(output, labels)

            # accumlate the loss
            accum_train_loss += loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        accum_val_loss = 0
        with torch.no_grad():
            for x, (imgs, labels) in enumerate(val_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                accum_val_loss += loss_function(output, labels).item()
        
        # print statistics of the epoch
        train_loss = accum_train_loss / n
        val_loss = accum_val_loss / x

        if(patience == 0):
            break
        elif(val_loss > prev_val_loss):
            patience = patience - 1
        else:
            best_model_at_epoch = epoch
            best_model = model
            prev_val_loss = val_loss
            patience = HP["patience"]
        print(f'Train Loss = {train_loss:.4f}\tVal Loss = {val_loss:.4f}\tPatience: {patience}')

    return [HP["neurons"][m],HP["hidden_layers"][i],HP["lr"][j],HP["activation_funcs"][k],best_model_at_epoch, train_loss,val_loss],best_model


def test(_model):
    # Compute Test Accuracy
    _model.eval()
    with torch.no_grad():
        correct = total = 0
        for images, labels in tqdm(test_loader,total=len(test_loader), desc=f"Test: "):
            images, labels = images.to(device), labels.to(device)
            output = _model(images)
            
            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0)

    print(f'Test Accuracy = {100 * correct/total :.3f}%')
    return 100 * correct/total


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


    train_loader = DataLoader(train_set, batch_size=HP["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=HP["batch_size"])
    val_loader = DataLoader(val_set, batch_size=HP["batch_size"])

    results = []
    
    best_model = None
    c= 0
    for m in range(len(HP["neurons"])):
        for i in range(len(HP["hidden_layers"])):
            for j in range(len(HP["lr"])):
                for k in range(len(HP["activation_funcs"])):
                    print(f"----- Model: {c} -----")
                    model = MyModel(10, hidden_layers=((np.array(HP["hidden_layers"][i])*HP["neurons"][m]).tolist()),activation=HP["activation_funcs"][k]).to(device)
                    loss_function = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=HP["lr"][j])
                    a,best_model = fit(m,i,j,k)
                    a.append(test(best_model))
                    print(a)
                    results.append(a)
                    c=c+1
                
    print(results)
    fields= ["neurons","hidden_layers","lr","activation_funcs","best_epoch","train_loss","val_loss","test_accuracy"]
    import csv
    with open('RESULTS', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(results)