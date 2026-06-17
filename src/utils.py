import matplotlib.pyplot as plt

def print_accuracy(model, x_test, y_test):
    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Test Accuracy: {accuracy:.4f}")

def plot_history(history, title="Training History"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f"{title} - Loss Plot")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title(f"{title} - Accuracy Plot")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_value_per_epoch(value_list, title="Value Per Epoch"):
    plt.figure(figsize=(6, 4))
    plt.plot(value_list, label='Epoch Value')
    plt.title(f"{title} - Value Plot Per Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

import os
import sys

def set_high_priority():
    # For Windows
    if sys.platform == 'win32':
        import win32process
        import win32api
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32process.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
    
    # For Mac / Linux
    else:
        # 0 is default, negative numbers mean higher priority (up to -20)
        # -20 completely prioritizes the process but might lock up your mouse/UI
        try:
            os.nice(-10) 
        except PermissionError:
            print("Run script with 'sudo' on Mac/Linux to force higher CPU priority.")



__all__ = [
    "print_accuracy",
    "plot_history",
    "plot_wall_clock_time_per_epoch",
    
]