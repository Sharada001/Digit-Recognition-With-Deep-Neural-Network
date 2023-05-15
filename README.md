# Digit Recognition With Deep Neural Network


  The project focuses on training a neural network for digit recognition using a dataset of 5000 handwritten images. The dataset is stored in a CSV file, with each image represented as a 20x20 matrix and labeled with the corresponding digit value.

<ul style="list-style-type: none"> 
<li>Dataset ~</li>
To process the dataset, Pandas and NumPy libraries are used. The images are shuffled randomly to ensure a balanced distribution of training and testing examples. The dataset is divided into a training set of 4000 instances and a testing set of 1000 instances.

<li>Neural Network Architecture ~</li>
The neural network architecture is implemented from scratch using NumPy. It allows customization of number of nodes in each layer and the number of layers. The output layer consists of 10 nodes, representing the 10 possible digits (0-9) for digit recognition.

 <li>Model Training ~</li>
The model is trained using gradient descent as the optimization technique. During training, forward propagation calculates the sigmoid values and activation outputs for each layer. Backpropagation computes the gradients for the weight parameters, considering the cost function and regularization to prevent overfitting.

 <li>Training Process ~</li>
The training process involves adjusting the weight parameters iteratively. The learning rate, regularization parameter, and number of iterations control the training process. The cost values are tracked to monitor the progress of the training.

 <li>Model Evaluation ~</li>
The trained model is evaluated using the testing dataset. Forward propagation is applied to the testing data using trained weights, obtaining sigmoid values and activation outputs. The predicted labels are determined by selecting the class index with the highest activation output. The accuracy is calculated by comparing the predicted labels with the true labels.

 <li>Saving the Model ~</li>
The trained weight parameters are saved using Pickle library, enabling the model to be loaded and reused for future tasks such as transfer learning or predictions on new data.
</ul>
<br>

<p align="center">
<img src="https://github.com/Sharada001/Digit-Recognition-With-Deep-Neural-Network/blob/0c10d2c73009fd4595b8eff6c77785e16ce0077e/Screenshots/nnn2%20(3).jpg" width="80%"/>
<img src="https://github.com/Sharada001/Digit-Recognition-With-Deep-Neural-Network/blob/0c10d2c73009fd4595b8eff6c77785e16ce0077e/Screenshots/nnn1%20(2).jpg" width="80%"/>
<img src="https://github.com/Sharada001/Digit-Recognition-With-Deep-Neural-Network/blob/0c10d2c73009fd4595b8eff6c77785e16ce0077e/Screenshots/nn_1%20(3).jpg" width="80%"/>
</p>
