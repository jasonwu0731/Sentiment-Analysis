## Use one layer-LSTM Encoder to classify sentiment

Runs the model on Twitter dataset.

### Requirements
Code is written in Python, Keras, sklearn and requires Tensorflow backend.


# Train the model: 
The trained parameters have been saved so you can test the model directly. But you can still re-trained it anyway.
P.S. for our own twitter dataset, it can achieve about 80+% accuracy.
```
$ python train.py
```

# Test the model:
```
$ python test.py
```

# Data Processing
one can use the function in `data_utils.py` to build own data token and dictionary
