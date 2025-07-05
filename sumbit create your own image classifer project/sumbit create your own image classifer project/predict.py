import argparse
import torch
from torchvision import models
from PIL import Image
import json
import numpy as np

# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(filepath):
    checkpoint = torch.load(filepath)    

    # Load the pre-trained model architecture
    model = models.__dict__[checkpoint['structure']](pretrained=True)
    
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = torch.optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optim_stat_dict'])

    epochs = checkpoint['epochs']

    return model, optimizer, epochs
    


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        im.rotate(30)
        im.resize((256,256))

        # Calculate the coordinates for the center crop
        output_size=224
        width, height = im.size
        left = (width - output_size) / 2
        top = (height - output_size) / 2
        right = (width + output_size) / 2
        bottom = (height + output_size) / 2

        # Crop the center portion of the image
        img_cropped = im.crop((left, top, right, bottom))        

        # # Horizontal flip the image
        img_flipped = img_cropped.transpose(Image.FLIP_LEFT_RIGHT)

        #color channels
        np_img=np.array(img_flipped)/255.0
        mean, std=np.array([0.485, 0.456, 0.406]),np.array([0.229, 0.224, 0.225])
        np_img=(np_img-mean)/std

        # #color channel first 
        np_img=np_img.transpose(2,0,1)

        tensor_img=torch.from_numpy(np_img).type(torch.FloatTensor)     
   
        return tensor_img

def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    img = process_image(image_path)

    
    with torch.no_grad():
        # logps = model.forward(img.to(device))
        logps = model.forward(img.unsqueeze(0))
        
    probabilities = torch.exp(logps).data
    
    return probabilities.topk(topk)



def print_predictions(args):
    # load model
    model, _, _ = load_model(args.model_filepath)

    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    if args.gpu and not(torch.cuda.is_available()):
        device = 'cpu'
        print('*'*50,"\nGPU was selected , but no GPU is available. Using CPU instead.")
    else:
        device = 'cpu'

    model = model.to(device)

    with open(args.category_names_filepath, 'r') as f:
        cat_to_name = json.load(f)
   
    ps = predict(args.image_filepath, model, device, args.top_k)    
    ps_values = ps.values.cpu().numpy()[0]

    class_labels = [str(i+1) for i in ps.indices.cpu().numpy()[0]]
    class_labels = [cat_to_name[str(i)] for i in class_labels]

    # Calculate the sum of all class probabilities
    sum_probabilities = sum(ps_values)

    # Calculate the probabilities as percentages
    probabilities = [(prob / sum_probabilities) * 100 for prob in ps_values]

    print('*'*50)
    print("Predictions:")
    print('-'*50)
    for i in range(args.top_k):
          print("#{: <3} {: <25} Prob: {:.2f}%".format(i+1, class_labels[i], probabilities[i]))
    
if __name__ == '__main__':
    
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(dest='image_filepath', help="path to the image file you want to classify")
    parser.add_argument(dest='model_filepath', help="path to the checkpoint file, including the extension")

    # optional arguments
    parser.add_argument('--category_names', dest='category_names_filepath', help="path to the json file that maps categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="This is the number of most likely categories to return, default is 5", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train the model on the GPU via CUDA", action='store_true')

    # Parse and print the results
    args = parser.parse_args()

    print_predictions(args)