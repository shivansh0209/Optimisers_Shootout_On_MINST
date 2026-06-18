##### In this file I write about the project journey of what I find new or what silly mistakes or valid mistakes I was doing while making the project.

1. I got very unstable values for the time per epoch plots it took me time to understand that it is not due to any flaw in the code, but it is due to some small level CPU disturbances. So then I came up with setting the priority of the training loop high for which I wrote a function and use the nice command to do so.

2. I didn't knew about the callbacks functionality of the Keras model class, which provided specific functions like on epoch end and on epoch start, et cetera. Using which I can write several callbacks.

3. I didn't knew that I can apply NAG momentum and simple SGD through the same SGD class.

4. When I studied the theory from some source, then I had a misconception that only one velocity variable will store the history of all the parameters, but there will be different storage for different parameters which was cleared when I tried to implement in the scratch ANN model

5. When I started to write notebook for learning rates, scheduler demonstration, then having a big neural network and a big dataset was problematic because the Kinks were not visible. For this, when I switched to a smaller database and simpler neural network, then I was able to demonstrate what I wanted to. 

6. I didn't knew that Python also contains a soft max in built function. Also, I use normal max function for applying the relu function but when performing with ML DL maths, one should always try to use the specific libraries built for them, even for the simplest tasks like I then used for calculating even the maximum.