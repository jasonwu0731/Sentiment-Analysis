## LSTM Model to classify sentiment
by Chien-Sheng (Jason) Wu
from Hong Kong University of Science and Technology(HKUST) Human Language Technology Center

### Requirements
Code is written in Python, Keras(>=2.0.5), sklearn and requires Tensorflow backend.

### Twitter Dataset with 1.6M Pos/Neg
Download [here](https://drive.google.com/drive/folders/0B_hiYftYF96RTmpxdTRkTV9DOTA?usp=sharing).

### Train the model: 
* For our own twitter dataset, it can achieve about 85+% accuracy in 3 epochs. 
```
$ python train.py
```

### Test the model:
Tesing the sentence input from user...
```
$ python test.py
```

### Data Processing
one can use the function in `data_utils.py` to build own data token and dictionary
