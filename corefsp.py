import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import models

# Versions should comply with PEP440.  For a discussion on single-sourcing
# the version across setup.py and the project code, see
# https://packaging.python.org/en/latest/single_source_version.html
__version__ = '0.1.0'


###########
# Callbacks
###########

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(tf.keras.callbacks.Callback, self).__init__()
        self._reset_history()

    def _reset_history(self):
        self.loss = []
        self.target_loss = []
        self.pwm_loss = []
        self.entropy_loss = []

    def on_train_begin(self, logs=None):
        self._reset_history()

    def on_batch_end(self, batch, logs=None):
        self.loss.append(logs.get('loss'))
        self.target_loss.append(logs.get('target_loss'))
        self.pwm_loss.append(logs.get('pwm_loss'))
        self.entropy_loss.append(logs.get('entropy_loss'))

class SequenceRecorder(tf.keras.callbacks.Callback):
    def __init__(self, loss_model, seq_filepath, batches):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.seq_filepath = seq_filepath
        self.batches = batches

        # Make generator model out of loss model
        seq_output = loss_model.get_layer("pwm_sample").output
        pred_output = loss_model.get_layer("target").input
        self.generator_model = models.Model(
            inputs=loss_model.input,
            outputs=[seq_output, pred_output]
        )

        # Clear destination file
        open(self.seq_filepath, 'w').close()
        
        # Store initial conditions
        self._store_sequences(-1)

    def _encode_seq(self, seqs_onehot):
        seqs = []
        for seq_onehot in seqs_onehot:
            seq = [['A', 'C', 'G', 'T'][np.where(n == 1)[0][0]] for n in seq_onehot]
            seq = ''.join(seq)
            seqs.append(seq)
        return seqs

    def _store_sequences(self, batch):
        # Extract sequences from model
        seq_vals, pred_vals = self.generator_model.predict([[0]])
        seqs = self._encode_seq(seq_vals)
        # Add sequences to file
        with open(self.seq_filepath, "a") as f:
            f.write('\n'.join([f"{batch+1}\t{s}" for s in seqs]) + '\n')
        return

    def on_batch_end(self, batch, logs=None):
        if isinstance(self.batches, int):
            if (batch + 1) % self.batches == 0:
                self._store_sequences(batch)
        else:
            if (batch + 1) in self.batches:
                self._store_sequences(batch)

########################
# Functions for sampling
########################

@tf.custom_gradient
def st_hardmax(logits):
    """TODO: docstring"""
    def grad(upstream):
        return upstream
    onehot_dim = logits.get_shape().as_list()[1]
    sampled_onehot = tf.one_hot(tf.argmax(logits, 1), onehot_dim, 1.0, 0.0)

    return sampled_onehot, grad

def max_pwm_st(pwm_logits):
    """TODO: docstring"""
    n_channels = tf.shape(pwm_logits)[-1]
    seq_length = tf.shape(pwm_logits)[-2]
    n_seqs = tf.shape(pwm_logits)[-3]

    flat_pwm = tf.reshape(pwm_logits, (n_seqs * seq_length, n_channels))
    sampled_pwm = st_hardmax(flat_pwm)

    return tf.reshape(sampled_pwm, (n_seqs, seq_length, n_channels))

###############################
# Main sequence design function
###############################

def design_seqs(
        predictor_model,
        target_loss_func,
        seq_length,
        n_seqs=1,
        target_weight=1,
        pwm_loss_func=None,
        pwm_weight=1,
        entropy_weight=1,
        learning_rate=0.1,
        n_iter_max=2000,
        seq_record_filename=None,
        seq_record_batches=100,
    ):
    """
    TODO: docstring
    """

    # Obtain unspecified dimensions
    n_channels = predictor_model.layers[0].input.shape[-1]
    n_outputs = predictor_model.layers[-1].output.shape[-1]

    predictor_model.trainable = False

    # Define optimization network

    # Dummy input
    dummy_input = layers.Input(shape=(1,), name='dummy_input')

    # Embeddings containing logits to be optimized
    # Output should be (n_seqs, seq_length, n_channels)
    logits_embedding = layers.Embedding(
        1,
        n_seqs * seq_length * n_channels,
        embeddings_initializer='glorot_uniform',
        name='logits_embedding',
    )
    logits_flat = logits_embedding(dummy_input)

    logits_reshape = layers.Lambda(
        lambda x: tf.reshape(x, (n_seqs, seq_length, n_channels)),
        name='logits_reshape',
    )
    logits = logits_reshape(logits_flat)

    # Instance normalization
    logits_instnorm = layers.BatchNormalization(
        axis=[0, 2],
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        name='logits_instnorm',
    )
    logits_instnormed = logits_instnorm(logits)
    # logits_instnormed = logits

    # PWM normalization layer
    # Output should be (n_seqs, seq_length, n_channels)
    pwm_softmax = layers.Softmax(axis=-1, name='pwm_softmax')
    pwm = pwm_softmax(logits_instnormed)

    pwm_sample = layers.Lambda(lambda x: max_pwm_st(x), name='pwm_sample')
    # pwm_sample = layers.Lambda(lambda x: sample_pwm_gumbel(x), name='pwm_sample')
    sampled_pwm = pwm_sample(pwm)

    # Obtain predictions
    # Output should be (n_seqs, n_outputs)
    sampled_pred = predictor_model(sampled_pwm)

    # Calculate losses
    # Target loss
    get_target_loss = layers.Lambda(lambda x: target_loss_func(x), name='target')
    target_loss = get_target_loss(sampled_pred)

    # PWM loss
    if pwm_loss_func is None:
        def pwm_loss_func(pwm):
            return pwm*0

    get_pwm_loss = layers.Lambda(lambda x: pwm_loss_func(x), name='pwm')
    pwm_loss = get_pwm_loss(pwm)

    # Entropy loss
    def entropy_loss_func(pwm):
        """Loss that returns the mean entropy of the batch"""
        # Entropy for each base and each sequence
        entropy = - pwm * tf.math.log(K.clip(pwm, K.epsilon(), 1. - K.epsilon()) ) / tf.math.log(2.0)
        entropy = tf.reduce_sum(entropy, axis=-1)

        # Mean entropy across each sequence
        mean_entropy = tf.reduce_mean(entropy, axis=-1)

        # Mean entropy across the whole batch
        return tf.reduce_mean(mean_entropy)
        
    get_entropy_loss = layers.Lambda(lambda x: entropy_loss_func(x), name='entropy')
    entropy_loss = get_entropy_loss(pwm)

    # Define loss model
    loss_model = models.Model(
        [dummy_input],
        [target_loss, pwm_loss, entropy_loss],
    )

    # Compile loss model
    def get_weighted_loss(loss_coeff=1.):
        def _min_pred(y_true, y_pred):
            return loss_coeff * y_pred

        return _min_pred

    loss_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'target': get_weighted_loss(loss_coeff=target_weight),
            'pwm': get_weighted_loss(loss_coeff=pwm_weight),
            'entropy': get_weighted_loss(loss_coeff=entropy_weight),
        }
    )

    # loss_model.summary()

    # # Callbacks
    # early_stopping_callback = EarlyBatchStopping(
    #     min_delta=min_delta_loss_stop,
    #     min_batch=n_iter_min,
    # )
    loss_history_callback = LossHistory()
    callbacks = [loss_history_callback]

    if seq_record_filename is not None:
        seq_recorder_callback = SequenceRecorder(loss_model, seq_record_filename, seq_record_batches)
        callbacks.append(seq_recorder_callback)

    # Fit
    _ = loss_model.fit(
        [0]*n_iter_max,
        [[0, 0, 0]]*n_iter_max,
        epochs=1,
        batch_size=1,
        callbacks=callbacks,
    )

    # Build training history from history callback
    train_history = {}

    train_history['loss'] = np.array(loss_history_callback.loss)
    if target_weight != 0:
        train_history['target_loss'] = np.array(loss_history_callback.target_loss) / target_weight
    else:
        train_history['target_loss'] = np.array(loss_history_callback.target_loss)
    if pwm_weight != 0:
        train_history['pwm_loss'] = np.array(loss_history_callback.pwm_loss) / pwm_weight
    else:
        train_history['pwm_loss'] = np.array(loss_history_callback.pwm_loss)
    if entropy_weight != 0:
        train_history['entropy_loss'] = np.array(loss_history_callback.entropy_loss) / entropy_weight
    else:
        train_history['entropy_loss'] = np.array(loss_history_callback.entropy_loss)

    # Extract sampled sequences and model predictions
    seq_output = loss_model.get_layer("pwm_sample").output
    pred_output = loss_model.get_layer("target").input
    m = models.Model(
        inputs=loss_model.input,
        outputs=[seq_output, pred_output]
    )
    seq_vals, pred_vals = m.predict([[0]])

    return (seq_vals, pred_vals, train_history)
