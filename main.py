from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import os
from label import parse_fg_file
import numpy as np 
import math
import random 
from torchvision.transforms import ToTensor

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
            try:
                inputs.append(Image.open(filepath))
            except:
                print(f'Invalid file path {filepath}')
                return
            fg_file = identifier + '.fg'
            label = parse_fg_file(fg_file)
            label = torch.tensor(list(label['symmetric_shape_modes']) + list(label['asymmetric_shape_modes']) + list(label['symmetric_texture_modes']))
            labels.append(label)

    return inputs, labels

def get_max_label_value(labels):
    current_max = -math.inf
    for label in labels:
        for value in label:
            if abs(value) > current_max:
                current_max = abs(value)
    return current_max


def create_batches(data, batch_size):
    random.shuffle(data)

    batched_data = []
    for i in range(0, len(data), batch_size):
        
        batched_data.append(data[i:i+batch_size])
    return batched_data



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


def image_to_tensor(imgs):
    transform = ToTensor()
    imgs = [transform(img) for img in imgs]
    imgs = torch.stack(imgs, dim=0)
    return imgs

def train_model(directory):

    inputs, labels = load_images(directory)
    split_ratio = 0.8
    final_idx = round(len(inputs) * split_ratio)
    indices = list(range(len(inputs)))
    random.shuffle(indices)

    inputs_train = [inputs[i] for i in indices[:final_idx]]
    labels_train = [labels[i] for i in indices[:final_idx]]

    inputs_test = [inputs[i] for i in indices[final_idx:]]
    labels_test = torch.stack([labels[i] for i in indices[final_idx:]], dim=0)

    scaling_constant = get_max_label_value(labels)
    labels_test = labels_test

    epochs = 1000
    batch_size = 50

    resnet = InceptionResnetV1(pretrained='casia-webface')
    

    # Initialize MTCNN for face detection
    mtcnn = MTCNN()

    in_features = resnet.last_linear.in_features
    resnet.last_linear = torch.nn.Linear(in_features, 130)
    resnet.last_bn = torch.nn.BatchNorm1d(130)
    resnet.apply(reset_weights)

    

    initial_weights = resnet.last_linear.weight.data.clone()
    
    


    


    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        batched_inputs = create_batches(inputs_train, batch_size)
        batched_labels = create_batches(labels_train, batch_size)
 
        for i in range(len(batched_inputs)):

            optimizer.zero_grad()

            current_inputs = batched_inputs[i]
            print(image_to_tensor(current_inputs))
            print(image_to_tensor(current_inputs).shape)
      
       
            current_labels = torch.stack(batched_labels[i], dim=0) / scaling_constant
            aligned = mtcnn(current_inputs) 
            
            aligned = torch.stack(aligned) 
            aligned.requires_grad_(True)
            print(aligned.shape)
            embeddings = resnet(aligned)
            
            
            loss = loss_function(embeddings, current_labels)
            loss.backward()

            optimizer.step()
            print(loss.item())
        if epoch % 2 == 0:
            # After some training, check the weights
            trained_weights = resnet.last_linear.weight.data.clone()
            print(loss_function(initial_weights, trained_weights))
      
            # Compare the initial and trained weights for any layer
            if not torch.equal(initial_weights, trained_weights):
                print("Weights have been updated.")
            else:
                print("Weights have not changed.")
            # embeddings_end = resnet(aligned_og)


            # print(loss_function(embeddings_initial, embeddings_end))
            # print(embeddings_initial, end='\n\n\n')
            # print(embeddings_end)


train_model('Archive')
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