import torch 
from collections import deque 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  

from icecream import ic 

# TRAIN 
def train_cycle(train_dl: DataLoader, test_dl: DataLoader, model: torch.nn.Module, optimizer, params: dict)->None:
    loss_queue = []
    loss_deque = deque(maxlen=50) 
    for epoch in range(params['epochs']):
        loss = single_train(train_dl, model, optimizer)
        loss_deque.append(loss) 
        loss_queue.append(loss) 

        # Print Average every 10 epochs 
        if (epoch+1 )%10 == 0: 
            print(f'Epoch {epoch+1}\n\t-The loss is {loss}\n\t-The average loss int the deque is {sum(loss_deque)/len(loss_deque)}')
            test(test_dl,model)

        # Run Test and save Checkpoint 
        if (epoch+1)%100 == 0: 
            model.save_checkpoint(epoch,optimizer,params)
        
        # Artificial Learning rate Decay 
        if (epoch+1) % params["lr_decay_step_size"] == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = params["lr_decay_factor"] * param_group['lr'] 


    # Plot the loss over training time  
    plot_loss(loss_queue ,params)

    # Save the model 
    model.save_model(params)

def plot_loss(losses: list,params: dict)->None:
    # Plot results  
    plt.title("Loss over Training Time")
    plt.xlabel("Epochs x10")
    plt.ylabel("Loss")

    plt.plot(range(len(losses)), losses)

    if params["jupyter"]: plt.show
    else: plt.show()
 

def single_train(dl:DataLoader, model:torch.nn.Module, optimizer)->float:
    model.train()

    avg_loss= 0 
    for X,y in dl:
        y = y.type(torch.LongTensor) # LongTensor = Integer 
        optimizer.zero_grad()
    
        prediction = model(X)
        loss = model.loss(prediction, y)
        loss.backward()
        optimizer.step()
    
        avg_loss+= loss.item() 

    return avg_loss/len(dl) 

def test(test_dl:DataLoader, model:torch.nn.Module)->None:
    model.eval() 
    acc, count, test_loss  = 0, 0, []
    
    with torch.no_grad():
        for X,y in test_dl:
            y = y.long()
            y_pred = model(X)
            loss = model.loss(y_pred,y)
            test_loss.append(loss.item())
            acc += (torch.argmax(y_pred, 1) == y).float().sum()
            count += len(y)
    
    acc = acc/count
    print(f"\t-Test Loss: {(sum(test_loss)/len(test_loss))}") 
    print(f"\t-Model accuracy: {(acc*100)}") 

