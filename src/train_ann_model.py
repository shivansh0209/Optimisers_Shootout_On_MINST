from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from src.time_history import TimeHistory
from src.utils import set_high_priority
import tensorflow as tf
import numpy as np
import random

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)



def train_model(x_train, y_train, optimiser, use_batchnorm = True, learning_rate_scheduler = None, dropout_rate = 0.3, epochs = 20, batch_size = 128):
    set_high_priority()
    input_size  =  x_train.shape[1]
    model  =  Sequential()
    model.add(Input(shape = (input_size,)))

    time_callback = TimeHistory()

    # Layer 1
    model.add(Dense(256, activation = 'relu'))
    if use_batchnorm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Layer 2
    model.add(Dense(128, activation = 'relu'))
    if use_batchnorm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Layer 3
    model.add(Dense(64, activation = 'relu'))
    if use_batchnorm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(10, activation = 'softmax'))
    
    # Using a slightly lower learning rate for smoother convergence
    model.compile(
        optimizer = optimiser, 
        loss = 'sparse_categorical_crossentropy', 
        metrics = ['accuracy']
    )
    
    model.summary()
    
    # Early stopping prevents the model from continuing to overfit once validation loss stalls
    early_stopping = EarlyStopping(
        monitor = 'val_loss', 
        patience = 4, 
        restore_best_weights = True
    )

    history  =  model.fit(
        x_train, y_train, 
        validation_split = 0.2, 
        epochs = epochs, 
        batch_size = batch_size, 
        callbacks = [early_stopping, time_callback, learning_rate_scheduler] if learning_rate_scheduler else [early_stopping, time_callback],
        verbose = 0
    )

    return model, history, time_callback.times

__all__  =  ['train_model']