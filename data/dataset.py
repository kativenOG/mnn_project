import os,torch 
import numpy as np
from PIL import Image

import torchvision.transforms.v2 as transforms 
from torch.utils.data import random_split 
from torch.utils.data import TensorDataset, DataLoader

def to_categorical(y, num_classes):
    appo = np.eye(num_classes, dtype='uint8')[y]
    return np.array([item[0] for item in appo])

def get_img(data_path, img_size):
    # Getting image array from path:
    img = Image.open(data_path)
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    return img

def get_dataloaders(shuffle:bool =True, batch_size:int = 32,
                dataset_path: str='data/Dataset_sign_language',
                img_size: int=64,
                test_size: float=0.15, 
                device:torch.device= torch.device('cpu'),
                grayscale = False )-> tuple[DataLoader,DataLoader]:

    print("Fetching the Data...")

    try: # Load npy files and transform them to torch Tensors  
        X = np.load(os.path.join(dataset_path,'X.npy'))
        Y = np.load(os.path.join(dataset_path,'Y.npy'))
        X,Y = torch.from_numpy(X).to(torch.float32).to(device), \
              torch.from_numpy(Y).to(torch.long).to(device)
    except: # Generate npy files if they are not there 
        # Get Data:
        labels = os.listdir(os.path.join(dataset_path,'Dataset')) # Geting labels
        X,Y = [],[]
        for i, label in enumerate(labels):
            datas_path = os.path.join(dataset_path,'Dataset',label)
            for data in os.listdir(os.path.join(datas_path)):
                img = get_img(os.path.join(datas_path,data), img_size)
                X.append(img)
                Y.append(i)

        # Create dateset:

        transform = transforms.Compose([ transforms.PILToTensor() ]) 
        X = [transform(x) for x in X]
        X = torch.stack(X).to(torch.float32).to(device)

        # Transform Labels
        Y = np.array(Y).astype(np.int8)
        num_class = len(set(Y.tolist()))
        Y = to_categorical(Y, num_class)
        Y = torch.from_numpy(Y).to(torch.long).to(device)

        if not os.path.exists(dataset_path): os.makedirs(dataset_path)
        np.save(os.path.join(dataset_path,'X.npy'), X.detach().cpu().numpy())
        np.save(os.path.join(dataset_path,'Y.npy'), Y.detach().cpu().numpy())


    # GrayScale Augmentation 
    if grayscale:
        transform = transforms.Compose([ transforms.Grayscale() ]) 
        X = transform(X)

    # Dataset and Dataloder Generation 
    print("Generating DataLoaders...")
    full_dataset= TensorDataset(X,Y)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
 
    train_dl, test_dl = DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size), \
                        DataLoader(test_dataset,shuffle=shuffle,batch_size=batch_size)  
     
    print("Finished!\n")
    return train_dl, test_dl
