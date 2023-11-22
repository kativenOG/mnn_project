# Project: Sign Language Digit Recognition
Project for Mathematics of Neural Networks 2MMA80, CNN for the classification task **Dataset_sign_language**.
## Grading
**DUE DATE 18th of January**.
The project is graded /10 with points given for:

| Item | Points |
|:----------------------------------------------------------------------------------------------------|--------|
| Correctly set up the training process with split dataset, batched SGD, etc.                         |   2.0  |
| Define and train a neural network that can classify the given images                                |   2.0  |
| Document every step you take, <br>others should understand what is happening from the text/comments |   3.0  |
| Describe and perform an experiment of your choice, <br>report your results and draw conclusions     |   3.0  |

## Extras: 

Choose one of these:
- Vary the width/depth
- Compare different activation functions
- Try data augmentation
- Compare different SGD variants
- Try different loss functions
- Compare different initialization schemes (such as found in [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html))
- Use the full sized color images and compare the difference with the grayscale images
- Can the network recognized rotated images?
- Analyse the evolution of the parameters during training.

## Dependencies
To run the code we need to install the python Virtual Environment with all the dependencies. <br/>
Install venv on your machine:
```
sudo apt update && sudo apt upgrade
sudo install python3-venv
```
Then clone the repo and run the installation script:
```
git clone https://github.com/kativenOG/mnn_project
cd  mnn_project/deps
bash installer.sh 
```
In a new terminal just run:
```
mnn
python3 main.py
```
Or run the code through the notebook using VSCode (or any other iPython environment):
```
mnn
code project.ipynb
```

## Todo:
- [ ] Add initialization scheme;
- [ ] Add more metrics to the Test;
- [ ] Choose one or more Extras;
- [ ] Write the project Notebook explaining every step.
