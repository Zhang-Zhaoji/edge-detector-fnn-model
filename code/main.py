import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

class FCNN(nn.Module):
    def __init__(self, in_channels=25, hidden_channels_1=32,hidden_channels_2=16, out_channels=2) -> None:
        super().__init__()
        # inchannels: dimenshion of input data = 5*5 = 25
        # hidden_channels：32->16
        # out_channels: number of categories. In this case is on or off edge
        self.model = nn.Sequential(
            # full connected layer
            nn.Linear(in_channels,hidden_channels_1),
            # activation function
            nn.ReLU(),
            # full connected layer
            nn.Linear(hidden_channels_1,hidden_channels_2),
            nn.ReLU(),
            # full connected layer
            nn.Linear(hidden_channels_2,out_channels)
        )
    def forward(self, x: torch.Tensor): 
        x = torch.flatten(x,1).float()
        return self.model(x)

class Edge_Dataset(Dataset):
    def __init__(self,X_path,Y,transform = None) -> None:
        # X, Y for image paths and labels, where $label \in \{0,1\}$ 
        super().__init__()
        self.X_path = X_path
        Y_dataset = np.zeros((len(Y),2))
        Y_dataset[:,0] += np.array(Y)
        Y_dataset[:,1] = np.ones(len(Y))-Y_dataset[:,0]
        self.Y = Y_dataset
        self.transform = transform

    def __len__(self):
        return len(self.X_path)
    
    def __getitem__(self, index) -> tuple:
        image = PIL.Image.open(self.X_path[index])
        if self.transform:
            image = self.transform(image)
        return (image,self.Y[index])

class Graph_Edge_Dataset(Dataset):
    def __init__(self,X_path,Y,transform = None) -> None:
        # X, Y for image paths and labels, where $label \in \{0,1\}$ 
        super().__init__()
        self.X_path = X_path
        Y_dataset = np.zeros((len(Y),2))
        self.Y = Y_dataset
        self.transform = transform

    def __len__(self):
        return len(self.X_path)
    
    def __getitem__(self, index) -> tuple:
        image = PIL.Image.open(self.X_path[index])
        if self.transform:
            image = self.transform(image)
        return (image,self.Y[index])

def train(model = FCNN(in_channels=25,hidden_channels_1=32,hidden_channels_2=16,out_channels=2)):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: SVM loss of Cross-entropy loss
        optimizer: Adamw or SGD
        scheduler: step or cosine
        args: configuration
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Loading data!")
    code_path = os.getcwd()
    train_path = os.path.dirname(code_path) + "\\traindata\\"
    # I put all data together
    data_path = train_path + "4\\"
    data_paths = []
    with open(train_path+"on_edge.txt","r") as file1:
        data_paths = file1.readlines()
    data_paths = list(map(lambda x: data_path + x, data_paths))
    on_length = len(data_paths)
    with open(train_path+"off_edge.txt","r") as file2:
        off_data_paths = file2.readlines()
    off_data_paths = list(map(lambda x: data_path + x, off_data_paths))
    off_length = len(off_data_paths)
    labels = [1]*on_length + [0] * off_length
    labels = np.array(labels)
    data_paths += off_data_paths
    data_paths = list(map(lambda x: x.replace("\n",""),data_paths))
    # make it the same for prediction task
    transform = transforms.Compose(
    [transforms.Grayscale(1),
     transforms.ToTensor()])
    # set a seed to repeat the training process
    np.random.seed(114514)
    train_labels = np.random.binomial(1,0.8,size=len(data_paths))
    train_data_path = []
    valid_data_path = []
    train_data_label = []
    valid_data_label = []
    for index in range(train_labels.shape[0]):
        if train_labels[index] == 1:
            train_data_path.append(data_paths[index])
            train_data_label.append(labels[index])
        else:
            valid_data_path.append(data_paths[index])
            valid_data_label.append(labels[index])
    # creating dataset
    edgedataset = Edge_Dataset(train_data_path,train_data_label,transform=transform)
    valid_dataset = Edge_Dataset(valid_data_path,valid_data_label,transform=transform)
    # setting hyper parameters
    batch_size = 16
    loss_function = torch.nn.functional.cross_entropy
    optimizer = torch.optim.AdamW(model.parameters(),lr = 0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

    edge_dataloader = DataLoader(dataset=edgedataset,batch_size=batch_size,shuffle=True, num_workers=8)
    edge_dataloader
    valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True, num_workers=8)
    print("Data loaded successfully!")
    print("start training!")
    # for-loop 
    num_epochs = 50
    train_acc = []
    test_acc = []
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_correct = 0
        train_total = 0
        print(f"training {epoch}times")
        # train
        _i = 0
        running_loss = 0
        for data in tqdm(edge_dataloader):
            #break
            _i += 1
            # get the inputs; data is a list of [inputs, labels]
            X, y = data
            X = X.to(device)
            y = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model.forward(X)
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y[:,1]).sum().item()
            # loss backward
            loss = loss_function(outputs,y)
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
            train_loss += loss.item()
            if _i % 1000 == 999:    # print every 1000 mini-batches
                print(f'[{epoch + 1}, {_i + 1:5d}] loss: {running_loss / 1000/batch_size:.7f}')
                running_loss = 0.0
        # adjust learning rate
        scheduler.step()
        # test
        test_loss, correct, total = test(test_dataloader=valid_dataloader,inheritage_model=model)
        print(f"train Loss = {train_loss/len(valid_data_label):.5f}")
        print(f'Accuracy of the network on the {len(train_data_path)} test images: {100 * train_correct / train_total:.5f} %')
        print(f"test Loss = {test_loss/len(valid_data_label):.5f}")
        print(f'Accuracy of the network on the {len(valid_data_path)} test images: {100 * correct / total:.5f} %')
        train_acc.append(100 * train_correct / train_total)
        train_losses.append(train_loss/len(valid_data_label))
        test_losses.append(test_loss/len(valid_data_label))
        test_acc.append(100 * correct / total)
            # forward
        PATH = f'./{epoch+1}_no_regu_epoch_edge_detector.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
    print(f"train_acc:{train_acc}")
    print(f"train_loss:{train_losses}")
    print(f"test_acc:{test_acc}")
    print(f"test_loss:{test_losses}")
    # write a file to show traing data
    with open('edge_example.txt', 'w') as file:
        file.write("train_acc,train_loss,test_acc,test_loss:\n")
        for i in range(num_epochs):
            file.write(f"{train_acc[i]},{train_losses[i]},{test_acc[i]},{test_losses[i]}\n")


def test(model_path = None, test_dataloader = None, inheritage_model = None, model = FCNN(in_channels=25,hidden_channels_1=32,hidden_channels_2=16,out_channels=2)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.functional.cross_entropy
    batch_size = 16
    print("loading model......")
    if inheritage_model != None:
        model = inheritage_model.to(device)
    else:
        if model_path == 'None':
            raise RuntimeError("NO model_path input while there are no inhertage model!")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
    print("model loading successfully!")

    print("loading test_dataloader......")
    if test_dataloader != None:
        testloader = test_dataloader
        print("inheritage test dataloader successfully!")
    else:
        print("Loading data!")
        #主要是我们实际上没有真正意义上的测试数据，我们就只是在这里验证一下在给定test dataloader或者全数据集上的performance
        code_path = os.getcwd()
        train_path = os.path.dirname(code_path) + "\\traindata\\"
        # I put all data together
        data_path = train_path + "4\\"
        data_paths = []
        with open(train_path+"on_edge.txt","r") as file1:
            data_paths = file1.readlines()
        data_paths = list(map(lambda x: data_path + x, data_paths))
        on_length = len(data_paths)
        with open(train_path+"off_edge.txt","r") as file2:
            off_data_paths = file2.readlines()
        off_data_paths = list(map(lambda x: data_path + x, off_data_paths))
        off_length = len(off_data_paths)
        labels = [1]*on_length + [0] * off_length
        data_paths += off_data_paths
        data_paths = list(map(lambda x: x.replace("\n",""),data_paths))

        transform = transforms.Compose(
        [transforms.Grayscale(1),
        transforms.ToTensor()])
        testset = Edge_Dataset(data_paths,labels,transform=transform)   
        testloader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=True, num_workers=8)
        print("testloader created successfully!")
    test_loss = 0
    total = 0
    correct = 0
    print("examine......")
    with torch.no_grad():
            for _data in testloader:
                # forward
                _X,_y = _data
                _X = _X.to(device)
                _y = _y.to(device)
                test_outputs = model.forward(_X)
                test_loss += loss_function(test_outputs,_y)
                _, predicted = torch.max(test_outputs.data, 1)
                total += _y.size(0)
                correct += (predicted == _y[:,1]).sum().item()
    print("test finished!")
    return test_loss, correct, total
    

    
def predict(graph_path , model_path, model = FCNN(in_channels=25,hidden_channels_1=32,hidden_channels_2=16,out_channels=2)):
    # load model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    """make graph list"""
    graph_Paths = get_file_names(graph_path)
    print(graph_Paths)
    graph_paths = list(map(lambda x: graph_path +"\\"+ x, graph_Paths))
    # transform to gray image
    transform = transforms.Compose(
    [transforms.Grayscale(1),
     transforms.ToTensor()])
    graph_list = []
    for path in graph_paths:
        image = PIL.Image.open(path)
        image = transform(image)
        graph_list.append(image)
    pad_function = nn.ReflectionPad2d(2)
    for index in trange(len(graph_list)):
        # for every single graph
        graph = graph_list[index]
        name = graph_Paths[index].split(".")[0]
        channel_, x, y = graph.size()
        output = 0 * graph
        # padding
        padded_graph = pad_function(graph)
        for k in range(2,2+x):
            for j in range(2,2+y):
                predicted_vector = model(padded_graph[:,k-2:k+3,j-2:j+3])
                output[:,k-2,j-2] = 1 if predicted_vector[:,0].item()>0 else 0
        plt.imsave(f"graph2/{name}.jpg",output[0,:,:])

def get_file_names(directory):
    file_names = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            file_names.append(file)
    return file_names

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='None')
    parser.add_argument('--graph_path', type=str, default='None')
    args = parser.parse_args()

    # create model
    if args.run == 'train':
        train()
    elif args.run == 'test':
        test(model_path=args.model_path)
    elif args.run == "predict":
        if args.graph_path != 'None' and  args.model_path != 'None':
            predict(args.graph_path, args.model_path)
        else:
            predict("../test_images/synthetic_characters", "./50_no_regu_epoch_edge_detector.pth")
      
    else: 
        raise AssertionError