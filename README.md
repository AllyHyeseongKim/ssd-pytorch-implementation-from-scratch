## SSD implementation from scratch

ssd detection implementation from scratch 

### development step

1. dataset : voc 2012 ++ voc 2007 dataset
2. network : we use `pre-trained` fc-reduced VGG 16 PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth


### file structure

```bash
root|-- dataset
        |-- __init__.py
        |-- dataset.py
    |-- model
        |-- __init__.py
        |-- model.py
    |-- learning
        |-- __init__.py
        |-- evaluator.py
        |-- optimizer.py
    |-- .gitignore
    |-- data
        |-- train.py
        |-- test.py
    |-- logs
    |-- save
    |-- train.py
    |-- test.py
    |-- utils.py
```
