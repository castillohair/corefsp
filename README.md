# Fast SeqProp: gradient-based sequence design

Core functionality from [Fast SeqProp](https://github.com/johli/seqprop), now compatible with tensorflow 2.

## Installation

Download or clone the repo and use ``python setup.py install``.

## Usage

The following code loads a pretrained predictor and generates 100 sequences to maximize its output.

```
import tensorflow as tf
from tensorflow.keras import models

import corefsp

predictor = models.load_model('model_filename.h5')

# Loss function
def target_loss_func(y_pred):
    return - tf.reduce_sum(y_pred)

# Generate sequences
# seq_vals has shape (n_seqs, seq_length, n_channels)
# and contains one hot-encoded designed sequences.
# pred_vals contains predictions for each generated sequence.
# train_history is a dictionary with keys 'loss', 'target_loss', and
# 'entropy_loss', containing loss values for each iteration.
seq_vals, pred_vals, train_history = corefsp.design_seqs(
    predictor,
    target_loss_func,
    seq_length=500,
    n_seqs=100,
    target_weight=1,
    learning_rate=0.1,
    n_iter_max=1000,
)

```