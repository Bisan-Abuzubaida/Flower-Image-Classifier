import argparse
import torch
from torchvision import datasets, models, transforms
from torch import nn
from torch import optim
import os

def transformations(args):    

    train_dir = os.path.join(args.data_directory , 'train')
    valid_dir = os.path.join(args.data_directory , 'valid')

    # validate paths before doing anything else
    if not os.path.exists(args.data_directory):
        print(f"Data Directory doesn't exist: {args.data_directory}")
        raise FileNotFoundError
    if not os.path.exists(args.save_dir):
        print(f"Save Directory doesn't exist: {args.save_dir}")
        raise FileNotFoundError

    if not os.path.exists(train_dir):
        print(f"Train folder doesn't exist: {train_dir}")
        raise FileNotFoundError
    if not os.path.exists(valid_dir):
        print(f"Validation folder doesn't exist: {valid_dir}")
        raise FileNotFoundError

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(244),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
                                        
                                        ])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(244),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True,batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)


    return trainloader, validloader, train_dataset.class_to_idx

def train_model(args, trainloader, validloader, class_to_idx):
    
    selected_model=args.arch
    model = getattr(models, selected_model)(pretrained=True)
    # Get the last layer
    last_layer_model = list(model.children())[-1] 
    ##get input feature from first element in the last layer
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):        
            input_features = module.in_features
    
    # print(model)
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    ##get classifier architecture of pre-trained model (classifier, fc,..etc) 
    for key, value in model.named_children():
        if value == last_layer_model:
            last_layer_key = key

    setattr(model, last_layer_key, nn.Sequential(
                               nn.Linear(input_features,args.hidden_units),
                               nn.ReLU(),
                               nn.Dropout(.2),

                               nn.Linear(args.hidden_units, 1000),
                               nn.ReLU(),
                               nn.Dropout(.2),
                            
                               nn.Linear(1000,102),
                               nn.LogSoftmax(dim=1)))


    # start train model

    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    elif args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU was selected, but no GPU is available. so CPU is used instead.")
    else:
        device = 'cpu'
    print(f"Using {device} to train model.")

    model.to(device)

    train_losses=[]
    validation_losses=[]
    epochs=args.epochs
    for epoch in range(epochs):
        running_loss=0
        for images, labels in trainloader:
            images,labels= images.to(device),labels .to(device)
            optimizer.zero_grad()
            logps=model(images)
            loss=criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        else:
            valid_loss = 0
            valid_accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels =images.to(device), labels.to(device)
                    logps=model(images)
                    loss=criterion(logps, labels)
                    valid_loss+=loss.item()

                    ps=torch.exp(logps)

                    top_p , top_class=ps.topk(1, dim=1)
                    equals= top_class==labels.view(*top_class.shape)
                    valid_accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                
            train_losses.append(running_loss/len(trainloader))
            validation_losses.append(valid_loss/len(validloader))
            print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/len(trainloader):.3f}.. "
                    f"validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"validation accuracy: {valid_accuracy/len(validloader):.3f}")
            model.train()

# end train

# save model

    model.class_to_idx = class_to_idx
    checkpoint = {
                'structure': args.arch,
                'classifier': model.classifier,
                'epochs': args.epochs,
                'optim_stat_dict': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}

    # torch.save(checkpoint, os.path.join(args.save_dir, "checkpoint.pth"))
    torch.save(checkpoint, args.save_dir + "checkpoint.pth")
    print(f'model saved to {args.save_dir},"/","checkpoint.pth"')
    return True
    # end save

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # mandatory arguments
    parser.add_argument(dest='data_directory', help="This is the directory of the training images, Expect at least 2 folders within, 'train' & 'valid'")

    # optional arguments
    parser.add_argument('--save_dir', dest='save_dir', help="This is the directory where the checkpoints will be saved", default='saved_models')
    parser.add_argument('--learning_rate', dest='learning_rate', help="learning rate when training the model. Default is 0.001. Expect float type", default=0.001, type=float)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train the model on the GPU via CUDA", action='store_true')
    # parser.add_argument('--arch', dest='arch', help="Type of pre-trained model to use", default="vgg19", type=str, choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--arch', dest='arch', help="Type of pre-trained model to use", default="vgg19", type=str)
    parser.add_argument('--epochs', dest='epochs', help="Number of epochs. Default is 3. Expect int type", default=3, type=int)
    
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=4096)


    args = parser.parse_args()

    train_data_loader, valid_data_loader, class_to_idx = transformations(args)
    train_model(args, train_data_loader, valid_data_loader, class_to_idx)
    print("done")
    
    