import torch  

# Lib Imports  
from utils.dataset import get_dataloaders 
from model.model import CNN  
from model.train import train_cycle

if __name__=='__main__':
    params = {
                # Optimizer Hyperparams
                'lr': 0.00001,
                'lr_decay_factor': 0.5,
                "lr_decay_step_size": 500,

                # Standard hyperparams  
                'img_size': 64,
                'epochs':2000,
                'params_dir':'params'
            }

    # DEVICE 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # DATA 
    train_dl, test_dl = get_dataloaders(img_size=params['img_size'],device=device)

    # MODEL 
    model = CNN(n_in=3,k=3 ,fc_hidden=15, n_classes=10).to(device)

    # OPTIMIZER 
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=params['lr'])

    # TRAIN AND TEST MODEL 
    train_cycle(train_dl,test_dl,model,optimizer,params)

