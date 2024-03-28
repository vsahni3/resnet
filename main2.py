from facenet_pytorch import InceptionResnetV1, MTCNN
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from label import parse_fg_file
import numpy as np 
import math
import random 
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, scaling_constant, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # Assuming this is already a tensor of shape (num_samples, 130)
        self.transform = transform
        self.scaling_constant = scaling_constant

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx])

        # Apply transformations (e.g., converting to tensor)
        if self.transform:
            image = self.transform(image)

        # Get the corresponding label tensor
        label = torch.tensor(self.labels[idx]) / self.scaling_constant

        return image, label


def loss_function(predictions, labels):
    loss = torch.nn.MSELoss()
    return loss(predictions, labels)

def load_images(directory):
    # map number to list of images
    inputs = []
    labels = []
    for filename in os.listdir(directory):
        # Create the full file path
        filepath = os.path.join(directory, filename)
        identifier = '_'.join(filepath.split('_')[:3])
        if filepath[-3:] != '.fg':
         
            inputs.append(filepath)
        
            fg_file = identifier + '.fg'
            label = parse_fg_file(fg_file)
            label = list(label['symmetric_shape_modes']) + list(label['asymmetric_shape_modes']) + list(label['symmetric_texture_modes'])
            labels.append(label)

    return inputs, labels

def get_max_label_value(labels):
    current_max = -math.inf
    for label in labels:
        for value in label:
            if abs(value) > current_max:
                current_max = abs(value)
    return current_max




def reset_weights(m):
    '''
    Resets model weights to default initialization.
    
    This function applies the default weight initialization
    method built into PyTorch layers. This includes layers like
    Linear, Conv2d, etc.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()



def train_model(directory, pretrained=False, model_path=''):

    inputs, labels = load_images(directory)

    scaling_constant = get_max_label_value(labels)

    # Define a transform to convert images to tensors and any other preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        # Add other transformations as needed, e.g., resizing, normalization
        transforms.Resize((224, 224)),  # Resize to a common size if your model requires it
    ])

    dataset = CustomDataset(inputs, labels, scaling_constant, transform=transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)  # 80% of the dataset for training
    test_size = dataset_size - train_size  # The remaining 20% for testing

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    epochs = 1500
    batch_size = 50

    # change model architecture

    resnet = InceptionResnetV1(pretrained='casia-webface')

    in_features = resnet.last_linear.in_features
    resnet.last_linear = torch.nn.Linear(in_features, 130)
    resnet.last_bn = torch.nn.BatchNorm1d(130)
    resnet.apply(reset_weights)

    if pretrained:
        optimizer, cur_epoch = load_partially_trained_model(model_path, resnet)
    else:
        optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)
        cur_epoch = 0

    running_loss = 0
    count = 0

    big_loss = 0
    big_count = 0


    

    for epoch in range(cur_epoch, epochs):
        
        
        for inputs, labels in train_loader:
       

            optimizer.zero_grad()
           
            

            embeddings = resnet(inputs) 
            
            loss = loss_function(embeddings, labels)
            loss.backward()

            optimizer.step()
            
            running_loss += loss.item()
            big_loss += loss.item()
            count += 1
            big_count += 1
        if epoch % 5 == 0:
            print(f'Average loss: {running_loss / count}')
            count = 0
            running_loss = 0
            # After some training, check the weights
            # trained_weights = resnet.last_linear.weight.data.clone()
            # print(loss_function(initial_weights, trained_weights))
      
            # # Compare the initial and trained weights for any layer
            # if not torch.equal(initial_weights, trained_weights):
            #     print("Weights have been updated.")
            # else:
            #     print("Weights have not changed.")
            # embeddings_end = resnet(aligned_og)


            # print(loss_function(embeddings_initial, embeddings_end))
            # print(embeddings_initial, end='\n\n\n')
            # print(embeddings_end)
    
        if (epoch + 1) % 100 == 0:
  
            checkpoint_filename = f'checkpoint_epoch_{epoch+1}.pth'
            
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': resnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': big_loss / big_count,
            }, checkpoint_filename)
            print(f"Saved checkpoint to {checkpoint_filename}")
            big_count = 0
            big_loss = 0

def load_partially_trained_model(model_path, model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Now, initialize the optimizer with the model's updated parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Load the optimizer state dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']  
    epoch = checkpoint['epoch']
    print(f'Current loss: {loss}') 
    return optimizer, epoch



train_model('Archive', True, 'checkpoint_epoch_500.pth')
# # Load pre-trained Inception ResNet model
# resnet = InceptionResnetV1(pretrained='casia-webface')

# # Initialize MTCNN for face detection
# mtcnn = MTCNN()
# # Load an image containing a face
# imgs = []
# imgs.append(Image.open('Archive/eastAsian_72_6e8e38bf13c1_0.75.jpg'))
# imgs.append(Image.open('Archive/african_60_5f6a3f05eea1_0.5.jpg'))


# # Preprocess the image and extract embeddings
# aligned = mtcnn(imgs)  
# # for i in range(len(aligned)):
# #     aligned[i] = torch.unsqueeze(aligned[i], 0)
# aligned = torch.stack(aligned)
# in_features = resnet.last_linear.in_features
# resnet.last_linear = torch.nn.Linear(in_features, 130)
# resnet.last_bn = torch.nn.BatchNorm1d(130)

# resnet.eval()
# embeddings = resnet(aligned).detach()
# # 'embeddings' now contains the feature vector for the detected face
# print(embeddings.shape)


    # sample_path1 = 'Archive/african_00_a99d642cde35_0.5.jpg'
    # sample_path2 = 'Archive/african_10_6f48b59b09df_0.5.jpg'
    # img1 = Image.open(sample_path1)
    # img2 = Image.open(sample_path2)
    # aligned_og = mtcnn([img1, img2]) 
    # aligned_og = torch.stack(aligned_og, dim=0)
    # embeddings_initial = resnet(aligned_og)
    # embeddings_end = resnet(aligned_og)