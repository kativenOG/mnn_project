# Standars:  

# My Stuff  
from utils.dataset import get_dataloaders 
from model.model import CNN  

if __name__=='main':
    img_size = 64
    train_dl, test_dl = get_dataloaders(img_size=img_size)
    CNN(img_size,)



