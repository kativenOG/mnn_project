import os,torch 
from typing import Tuple 
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def to_categorical(y, num_classes):
    appo = np.eye(num_classes, dtype='uint8')[y]
    return np.array([item[0] for item in appo])

def get_img(data_path, img_size, grayscale):
    # Getting image array from path:
    img = Image.open(data_path)
    img = img.convert("RGB")
    if grayscale: 
        values = []
        img_array = np.array(img).astype(np.float32)
        red,green,blue = img_array[:,:,0],img_array[:,:,1],img_array[:,:,2]
        values = (red + green + blue)/(256*3)
        # values = np.expand_dims(values,axis=2)
        img = Image.fromarray(values) 
    img = img.resize((img_size, img_size))
    return img

def get_dataset(dataset_path: str='Dataset_sign_language',img_size: int=64, grayscale: bool=False,test_size: float=0.2)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    try:
        X = np.load(os.path.join(dataset_path,'X.npy'))
        Y = np.load(os.path.join(dataset_path,'Y.npy'))
    except:
        # Get Data:
        labels = os.listdir(os.path.join(dataset_path,'Dataset')) # Geting labels
        X,Y = [],[]
        for i, label in enumerate(labels):
            datas_path = os.path.join(dataset_path,'Dataset',label)
            for data in os.listdir(os.path.join(datas_path)):
                img = get_img(os.path.join(datas_path,data), img_size, grayscale)
                X.append(img)
                Y.append(i)

        # Create dateset:
        X  = [np.array(x) for x in X]

        # X = 1-np.array(X).astype('float32')/255.
        X = np.array(X).astype('float32')
        Y = np.array(Y).astype(np.int8)
        num_class = len(set(Y.tolist()))
        Y = to_categorical(Y, num_class)

        if not os.path.exists(dataset_path): os.makedirs(dataset_path)
        np.save(os.path.join(dataset_path,'X.npy'), X)
        np.save(os.path.join(dataset_path,'Y.npy'), Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    X_train, X_test, Y_train, Y_test = torch.from_numpy(X_train),torch.from_numpy(X_test), \
                                       torch.from_numpy(Y_train),torch.from_numpy(Y_test)

    return X_train, X_test, Y_train, Y_test


def get_dataloaders(img_size:int = 64,
                    shuffle: bool = True,
                    grayscale: bool=False,
                    batch_size:int = 32)->Tuple[DataLoader, DataLoader]:
    X_train, X_test, Y_train, Y_test = get_dataset(img_size=img_size,grayscale=grayscale)

    # Cast to Tensor Dataset 
    train_dataset, test_dataset = TensorDataset(X_train,Y_train), \
                                  TensorDataset(X_test,Y_test) 

    # Create Dataloaders
    train_dl, test_dl = DataLoader(train_dataset,shuffle=shuffle,batch_size=batch_size), \
                        DataLoader(test_dataset,shuffle=shuffle,batch_size=batch_size)  
    
    return train_dl, test_dl


def visualize(grayscale=True)->None:
    import matplotlib.pyplot as plt  
    # Get data 
    X_train, X_test, Y_train, Y_test = get_dataset(grayscale=True)

    # Lets visualize some of the samples
    show = (100, 600, 1500, 1648)
    plt.figure(figsize=[4*len(show), 4])
    
    for i in range(len(show)):
        plt.subplot(1, len(show), i+1)
        plt.imshow(X_train[show[i]], cmap='gray_r')
        plt.axis("off")

    plt.show()
     
