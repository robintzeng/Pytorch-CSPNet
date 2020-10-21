import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping
from CSPResNet import csp_resnet152
import matplotlib.pyplot as plt

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(net,n_epoches = 500,
          patience =20,
          valid_size =  0.2,
          batch_size =  64):
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print(device)
    else: 
        device = torch.device("cpu")
        print(device)

    net.to(device)


    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)



    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size,
                                              sampler = train_sampler,
                                              num_workers=0)
    
    validloader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)


    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             num_workers=0)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    
   
    train_losses = []
    valid_losses = []
    test_acces = []
    avg_train_losses = []
    avg_valid_losses = []
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epoches + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        net.train()
        
        for i, data in tqdm(enumerate(trainloader, 0)):

            inputs, labels = data[0].to(device), data[1].to(device)
            
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            


            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        

        net.eval() # prep model for evaluation
        for data, target in validloader:
            # forward pass: compute predicted outputs by passing inputs to the model
            data, target = data.to(device), target.to(device)

            output = net(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: %d %%' % (
        test_acc))


        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        test_acces.append(test_acc)


        epoch_len = len(str(n_epoches))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epoches:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        train_losses = []
        valid_losses = []
        early_stopping(valid_loss, net)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        net.load_state_dict(torch.load('checkpoint.pt'))

    return avg_train_losses, avg_valid_losses,test_acces


if __name__ == "__main__":

    net = csp_resnet152(pretrained=False,num_classes = 10)
    #y = net(torch.randn(1, 3, 112, 112))
    #print(y.size())  
    train_loss, valid_loss,test_acces = train(net,n_epoches = 5,patience =20,valid_size =  0.2,batch_size =  64)
    
    fig = plt.figure()
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
    plt.plot(range(1,len(test_acces)+1),test_acces,label='Test accuracy')
    
    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 2.5) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')      
