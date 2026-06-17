import numpy as np
from scipy.special import softmax
import time

def training_two_layer_ann(x_train, y_train, n_hidden, output_size, beta = 0.9, learning_rate = 0.1, batch_size = 16, n_epochs = 1000):
    loss = []
    epoch_no = []
    time_per_epoch = []
    input_size = x_train.shape[1]
    
    rng = np.random.default_rng()
    wl1 = rng.standard_normal((input_size, n_hidden)) * np.sqrt(2. / input_size) # He initialization
    wl2 = rng.standard_normal((n_hidden, output_size)) * np.sqrt(2. / n_hidden)
    b1 = np.zeros((1, n_hidden))
    b2 = np.zeros((1, output_size))

    vwl1 = np.zeros_like(wl1)
    vwl2 = np.zeros_like(wl2)
    vb1 = np.zeros_like(b1)
    vb2 = np.zeros_like(b2)
    
    y_train_ohe = np.zeros((y_train.shape[0], output_size))
    for i in range(y_train.shape[0]):
        y_train_ohe[i][y_train[i]] = 1

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_start_time = time.time()

        for batch in range(0, x_train.shape[0], batch_size):
            x = x_train[batch: batch + batch_size]
            y = y_train_ohe[batch: batch + batch_size]
            actual_batch_size = x.shape[0]

            # Forward pass
            z1 = x @ wl1 + b1
            o1 = np.maximum(0, z1)

            z2 = o1 @ wl2 + b2
            o2 = softmax(z2, axis=1)

            batch_loss = -np.sum(y * np.log(o2 + 1e-15)) / actual_batch_size
            epoch_loss += batch_loss

            # Backward pass
            dz2 = o2 - y
            dwl2 = o1.T @ dz2 / actual_batch_size
            db2 = np.sum(dz2, axis=0, keepdims=True) / actual_batch_size

            do1 = dz2 @ wl2.T
            dz1 = do1 * (z1 > 0).astype(float)
            dwl1 = x.T @ dz1 / actual_batch_size
            db1 = np.sum(dz1, axis=0, keepdims=True) / actual_batch_size

            # Update weights
            vwl2 = beta * vwl2 + learning_rate * dwl2
            wl2 -= vwl2
            
            vwl1 = beta * vwl1 + learning_rate * dwl1
            wl1 -= vwl1
            
            vb2 = beta * vb2 + learning_rate * db2
            b2 -= vb2
            
            vb1 = beta * vb1 + learning_rate * db1
            b1 -= vb1

        # At the end of the epoch loop, before appending:
        batches = x_train.shape[0] / batch_size
        loss.append(epoch_loss / batches)
        epoch_no.append(epoch)
        epoch_end_time = time.time()
        time_per_epoch.append(epoch_end_time - epoch_start_time)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss[-1]:.4f}, Time: {time_per_epoch[-1]:.2f}s")

    return wl1, wl2, b1, b2, loss, epoch_no, time_per_epoch


def calculate_accuracy(wl1, wl2, b1, b2, x_test, y_test):
    z1 = x_test @ wl1 + b1
    o1 = np.maximum(0, z1)

    z2 = o1 @ wl2 + b2
    o2 = softmax(z2, axis=1)

    predictions = np.argmax(o2, axis=1)
    accuracy = np.mean(predictions == y_test)
    return accuracy

__all__ = ["training_two_layer_ann", "calculate_accuracy"]