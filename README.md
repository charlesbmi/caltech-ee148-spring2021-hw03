<div align="center">    
 
# EE148: MNIST Classification  

</div>
 
## Description   
Classify MNIST digits.

## How to run   
First, install dependencies  
```bash
# clone project   
git clone https://github.com/charlesincharge/caltech-ee148-spring2021-hw03

# install project (conda environment recommended) 
cd caltech-ee148-spring2021-hw03
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
