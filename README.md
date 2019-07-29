## yolo_v2_tensorflow

이 프로젝트는 voc 2007 + 2012 로 yolo_v2 실험을 해 보는 것입니다.

### attribute
1. task : detection

2. model : darknet-19 + (res net + cbam)

3. data : voc 2007 + voc 2012

4. loss : yolo loss

### experiment

epoch = 보통 100 

batch size = 10

init learning rate = 1e-4

learning rate decay = 0.5

learning rate patience = 10

eps = 1e-5

1. voc2007+voc2012 <br /> : yolo v2 eps 1e-8 <br />[mAP : 0.24, fps : 81] <br />160 epoch <br />(2019.01.03)

2. voc2007+voc2012+l2_reg <br /> : yolo v2 L2 Reg <br />[mAP : 0.24, fps : 81] <br />(2019.01.05)

3. voc2007+voc2012+ <br /> : yolo v2 + not L2Reg <br />[mAP : 0.58, fps : 81] <br />100 epoch <br />(2019.01.06)

4. voc2007+voc2012+ cbam <br /> : yolo v2 + cbam (모든 conv 뒤에 cbam 추가) <br />[mAP : 0.32, fps : 28] <br />100 epoch <br />(2019.01.07)

5. voc2007+voc2012+cbam++ <br /> : yolo v2 + cbam + resnet  <br />[mAP : 0.52, fps : 48.44] <br /> (bottle neck 에 residual connection 과 cbam 추가)  <br />
learning rate decay = 0.5  <br />learning rate patience = 20 <br /> (2019.01.08)

```buildoutcfg

# decide 4 of models. 
```

1. voc2007+voc2012+model_1 <br /> : yolo v2 <br /> [mAP : 0.64 fps : 79.93, recall : 0.22] 
<br /> learning rate decay = 0.5  <br />learning rate patience = 10 <br /> eps = 1e-5 <br /> (2019.01.12)

2. voc2007+voc2012+model_2 <br /> : yolo v2 + cbam <br /> [mAP : 0.26, fps : 48.07, recall : 0.19] 
<br /> learning rate decay = 0.5  <br />learning rate patience = 10 <br /> eps = 1e-5 <br /> (2019.01.14)

3. voc2007+voc2012+model_3 <br /> : yolo v2 + cbam + res<br /> [mAP : 0.63, fps : 47.26, recall : 0.24] 
<br /> learning rate decay = 0.5  <br />learning rate patience = 10 <br /> eps = 1e-5 <br /> (2019.01.11)

4. voc2007+voc2012+model_4 <br /> : yolo v2 + cbam + res + skip connection <br /> [mAP : 0.64, fps : 47.42, recall : 0.17] 
<br /> learning rate decay = 0.5  <br />learning rate patience = 10 <br /> eps = 1e-5 <br /> (2019.01.15)

### update

- mAP 기준 evaluation is_better 변경 

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
