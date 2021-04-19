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

# run module
python lit_image_classifier.py
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_image_classifier import LitImageClassifier
from pytorch_lightning import Trainer

# model
model = LitImageClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```
