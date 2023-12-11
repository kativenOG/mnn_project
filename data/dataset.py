from sklearn.model_selection import train_test_split
import os,torch 
import numpy as np
from PIL import Image

from icecream import ic 

import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader

class Dataset_sign_language(Dataset):

    def __init__(self, data: np.ndarray, y: np.ndarray, transform=None): 
        self.data = data
        self.label = torch.from_numpy(y).to(torch.int8)
        self.transform = transform 

    def __getitem__(self, index): 
        return self.transform(self.data[index]), self.label[index]
    
    def __len__(self): 
        return len(list(self.data)) 

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

    try: # Load npy files and transform them to torch Tensors  
        X = np.load(os.path.join(dataset_path,'X.npy'))
        Y = np.load(os.path.join(dataset_path,'Y.npy'))

        # Test :
        print(1)
        ic(X.shape,Y.shape)
        exit()
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

        # transform = transforms.Compose([ transforms.PILToTensor() ]) 
        X = [np.array(x) for x in X]
        X = np.stack(X).astype(np.float32)

        # Transform Labels
        Y = np.array(Y).astype(np.int8)
        num_class = len(set(Y.tolist()))
         
        ic(Y.shape)
        Y = to_categorical(Y, num_class)
        ic(Y.shape)
        exit()

        # Save the np.ndarray values 
        if not os.path.exists(dataset_path): os.makedirs(dataset_path)
        np.save(os.path.join(dataset_path,'X.npy'), X)
        np.save(os.path.join(dataset_path,'Y.npy'), Y)


    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=test_size) 

    # Data Augmentation 
    train_transform = [ 
            transforms.ToTensor(),
            transforms.RandomCrop(64, padding=2), 
            transforms.GaussianBlur(kernel_size=3)
        ]
    if grayscale: 
        train_transform.insert(1,transforms.Grayscale())
        train_transform.insert(2,transforms.Normalize(mean=(0.5),std=(0.5)))
    else: 
        train_transform.insert(2,transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5,0.5,0.5)))

    train_transform = transforms.Compose(train_transform) 
    train_dataset = Dataset_sign_language(x_train, y_train, train_transform)


    test_transform = [] 
    test_transform.append(transforms.ToTensor())
    if grayscale: 
        test_transform.insert(1,transforms.Grayscale())
        test_transform.insert(2,transforms.Normalize(mean=(0.5),std=(0.5)))
    else: 
        test_transform.insert(2,transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5,0.5,0.5)))
    test_transform = transforms.Compose(test_transform)
    test_dataset = Dataset_sign_language(x_test, y_test, test_transform)

    # Dataset and Dataloder Generation 
    train_dl, test_dl = DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size), \
                        DataLoader(test_dataset,shuffle=shuffle,batch_size=batch_size)  
     
    return train_dl, test_dl
