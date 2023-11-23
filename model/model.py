import torch,os,time
import torch.nn as nn  
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self,
                 # Conv params 
                 n_in:int,k:int, 
                 # MLP params 
                 fc_hidden: int, n_classes:int,
                 bn = False, img_size:int = 64)->None:

         
        super(CNN,self).__init__()

        # If we use Batch Normalization or not 
        self.batch_n = bn
        # Size of the Pool Kernel 
        self.pool_kernel = 5  

        #####Convolutional Layers and Batch Normalization Layers##### 

        # Initial Layer  
        self.conv_l, self.bn_l = [], []

        # Hidden layers 
        for _ in range(k):
            self.conv_l.append(nn.Conv2d(in_channels=n_in,out_channels=n_in,kernel_size=3)) 
            self.bn_l.append(nn.BatchNorm2d(n_in)) 

        # Transform to ModuleList
        self.conv_l = nn.ModuleList(self.conv_l)
        self.bn_l= nn.ModuleList(self.bn_l)
        
        # Pooling Function 
        self.pool = F.max_pool2d

        ##### Fully Connected Part #####
        # We just use one single Sequential
        self.fc_layers = nn.Sequential(
                nn.Flatten(), 
                nn.Linear(363,fc_hidden),
                nn.ReLU(), 
                nn.Linear(fc_hidden,fc_hidden), 
                nn.ReLU(),   
                nn.Linear(fc_hidden,n_classes), 
                nn.ReLU(),   
        ) 

    def forward(self,x):
        ######  Convolutional Part ######  
        for conv,bn in zip(self.conv_l,self.bn_l):
            x = conv(x)  
            if self.batch_n: x = bn(x)
            x = F.relu(x)

        ###### Pooling ######  
        x = self.pool(x,kernel_size=self.pool_kernel)

        ###### Fully connected Output ######  
        return nn.functional.softmax(self.fc_layers(x))

    def loss(self,x: torch.Tensor,y: torch.Tensor)-> torch.Tensor:
        return F.cross_entropy(x,y) 
    
    def save_model(self,params: dict):
        # Check if dir exists, if not make it  
        if not os.path.isdir(params['params_dir']): os.mkdir(params['params_dir'])
        # Save Model 
        print('Saving Model....')
        torch.save(self,os.path.join(params['params_dir'],'model_params.pt'))
        print('Model Saved!')

    def save_checkpoint(self,epoch:int,optimizer,params: dict):   
        # Check if dir exists, if not make it  
        if not os.path.isdir(params['params_dir']): os.mkdir(params['params_dir'])
        # Get timestamp 
        t = time.localtime()
        timestamp = time.strftime('%Y_%b_%d_%H_%M_%S', t)  
        # Save the params 
        print(f'Saving Checkpoint at Timestamp {timestamp}...')
        torch.save({
                'epoch': epoch,
                'model_state_dict':self.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
        },os.path.join(params['params_dir'],f'timestamp_{timestamp}.pth'))
        print('Checkpoint Saved!') 
    
def apply_initialization(m):
    if not isinstance(m, (nn.ModuleList)):
        if isinstance(m, (torch.nn.BatchNorm2d)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0.0001)
        elif isinstance(m, nn.Linear): 
            torch.nn.init.xavier_uniform_(m.weight) 
            m.bias.data.fill_(0.01)
        elif isinstance(m,nn.Conv2d):         
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    else:
        for val in m:
            if isinstance(val, (torch.nn.BatchNorm2d)):
                torch.nn.init.constant_(val.weight, 1)
                torch.nn.init.constant_(val.bias, 0.0001)
            elif isinstance(m, nn.Linear): 
                torch.nn.init.xavier_uniform_(val.weight) 
                val.bias.data.fill_(0.01) 
            elif isinstance(m,nn.Conv2d):         
                torch.nn.init.xavier_uniform_(val.weight) 

