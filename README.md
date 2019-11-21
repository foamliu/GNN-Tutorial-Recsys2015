# RecSys Challenge 2015

In this year’s edition of the RecSys Challenge, YOOCHOOSE is providing a collection of sequences of click events; click sessions. For some of the sessions, there are also buying events. The goal is hence to predict whether the user (a session) is going to buy something or not, and if he is buying, what would be the items he is going to buy. Such an information is of high value to an e-business as it can indicate not only what items to suggest to the user but also how it can encourage the user to become a buyer. For instance to provide the user some dedicated promotions, discounts, etc. The data represents six months of activities of a big e-commerce businesses in Europe selling all kinds of stuff such as garden tools, toys, clothes, electronics and much more.

A detailed description of the challenge can be found on the website of the [RecSys Challenge 2015](http://2015.recsyschallenge.com/).

## Dataset 
Please download the dataset [HERE](https://recsys.acm.org/recsys15/challenge/).

![image](https://github.com/foamliu/GNN-Tutorial-Recsys2015/raw/master/images/yoochoose-large.png)


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
