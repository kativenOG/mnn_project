import torch,json   

# Lib Imports  
from data.dataset import get_dataloaders 
from model.model import CNN,apply_initialization
from model.train import train_cycle

def read_params_file(base_params,file_name)->dict:
    """ 
    Get the content of the json file and merge it with the base Params 
    Json file contains the tuned params 
    """
    content = open(file_name).read()
    content = json.loads(content) 
    return {**base_params, **content}


if __name__=='__main__':
    params = {
                # Optimizer Hyperparams
                'lr': 0.0001,
                'lr_decay_factor': 0.5,
                "lr_decay_step_size": 300,

                # Standard hyperparams  
                'img_size': 64,
                'epochs':1000,

                # Other stuff 
                'params_dir':'params',
                'grayscale': True,
                'jupyter': False,
            }

    # DEVICE 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # DATA 
    train_dl, test_dl = get_dataloaders(img_size=params['img_size'],device=device,grayscale=params['grayscale'])
    

    # MODEL 
    n_channels = 3 if (not params['grayscale']) else 1 
    model = CNN(n_in=n_channels, k=3, fc_hidden=15, n_classes=10, bn=True,grayscale=params['grayscale']).to(device)
    model.apply(apply_initialization)

    # OPTIMIZER 
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=params['lr'])

    # TRAIN AND TEST MODEL 
    train_cycle(train_dl,test_dl,model,optimizer,params)

