import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#======= Load the Data =======
data = pd.read_csv("C://Users//Dylan//OneDrive//My Laptop//Desktop//MNIST.csv")

#Shuffle the dataset to ensure we get a random distribution of data for training and testing
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

#Separate the labels and the pixel values
labels = data.iloc[:, 0]  #First column is labels
pixels = data.iloc[:, 1:] / 255  #Remaining columns are pixel values, normalized to [0, 1]

#Split the dataset into training and testing sets (80% train, 20% test)
split = int(len(data) * 0.8)  #Calculate split index for 80/20 split

#Training set
xTrain = pixels[:split].to_numpy()
yTrain = labels[:split].to_numpy()

#Testing set
xTest = pixels[split:].to_numpy()
yTest = labels[split:].to_numpy()

#One-hot encode the training and testing labels
yTrain = np.eye(10)[yTrain]
yTest = np.eye(10)[yTest]

#======= Visualise the images ======
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]);
    plt.yticks([]);
    plt.grid(False);
    plt.imshow(xTrain[i].reshape(28,28), cmap=plt.cm.binary)

plt.show()

#======= Initialise Weights & Biases =======
#Network Shape = [784, 32, 32, 10]
W1 = np.random.randn(784, 32)
b1 = np.zeros((1, 32))
W2 = np.random.randn(32, 10)
b2 = np.zeros((1, 10))

#======= Activation Functions =======
def relu(x):
    return np.maximum(0, x)

def relu_prime(z):
    return np.where(z > 0, 1, 0)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#======= Loss & Accuracy =======
def loss(yTrue, yPred):
    m = yPred.shape[0]
    return -np.sum(yTrue * np.log(yPred + 1e-9)) / m

def accuracy(yTrue, yPred):
    predictions = np.argmax(yPred, axis=1)
    labels = np.argmax(yTrue, axis=1)
    return np.mean(predictions == labels)

#======= Forward Pass =======
def feedforward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

#======= Backward Pass =======
def backward(X, y, a1, a2, z1):
    delta2 = a2 - y
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    delta1 = np.dot(delta2, W2.T) * relu_prime(z1)
    dW1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

#======= Update Gradients =======
def update_gradients(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

#======= Mini-batch Gradient Descent =======
def mini_batch_gradient_descent(X, y, lr, batchSize):
    global W1, b1, W2, b2
    m = X.shape[0]
    indices = np.random.permutation(m)
    X = X[indices]
    y = y[indices]

    for i in range(0, m, batchSize):
        end = i + batchSize
        X_batch = X[i:end]
        y_batch = y[i:end]

        z1, a1, z2, a2 = feedforward(X_batch)
        dW1, db1, dW2, db2 = backward(X_batch, y_batch, a1, a2, z1)
        W1, b1, W2, b2 = update_gradients(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

#======= Training Function =======
def train(xData, yData, lr, batchSize, epochs):
    for epoch in range(epochs):
        mini_batch_gradient_descent(xData, yData, lr, batchSize)

        #Calculate and print the loss and accuracy after each epoch
        _, _, _, yPred = feedforward(xData)  # Using the entire dataset for evaluation
        trainLoss = loss(yData, yPred)
        trainAccuracy = accuracy(yData, yPred)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {trainLoss:.4f}, Accuracy: {trainAccuracy:.4f}")

train(xTrain, yTrain, lr=0.01, batchSize=64, epochs=100)

#======= Test Function =======
def test(xData, yData):
    _, _, _, yPred = feedforward(xData)
    testLoss = loss(yData, yPred)
    testAccuracy = accuracy(yData, yPred)
    print(f"Test Loss: {testLoss:.4f}, Test Accuracy: {testAccuracy:.4f}")

test(xTest, yTest)
      
def incorrectPreds(xTest, yTest):
  _, _, _, yPred = feedforward(xTest)
  incorrect_indices = np.where(np.argmax(yTest, axis=1) != np.argmax(yPred, axis=1))[0]

  for i in range(10):
    idx = incorrect_indices[i]
    plt.figure()
    plt.imshow(xTest[idx].reshape(28, 28), cmap=plt.cm.binary)
    predicted_label = np.argmax(yPred[idx])
    actual_label = np.argmax(yTest[idx])
    plt.title(f"Predicted: {predicted_label}, Actual: {actual_label}")
    plt.xticks([]);
    plt.yticks([]);
    plt.show()

incorrectPreds(xTest, yTest)