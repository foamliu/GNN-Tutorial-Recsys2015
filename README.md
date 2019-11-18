# RecSys Challenge 2015

Given a sequence of click events performed by some user during a typical session in an e-commerce website, the goal is to predict whether the user is going to buy something or not, and if he is buying, what would be the items he is going to buy. 

## Dataset 
Please download the dataset [HERE](https://recsys.acm.org/recsys15/challenge/).

## Dependencies

- Python 3.6.8
- PyTorch 1.3.0

## Usage
### Data Pre-processing
Extract training images:
```bash
$ python extract.py
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

### Performance

|epoch|Loss|Train Auc|Val Auc|Test Auc|
|---|---|---|---|---|
|0|0.20631|0.77606|0.73532|0.73023|
|1|0.18863|0.80737|0.73994|0.73288|
