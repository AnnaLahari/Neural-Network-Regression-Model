# Developing a Neural Network Regression Model

## AIM :

To develop a neural network regression model for the given dataset.

## THEORY :

The class NeuralNet inherits from nn.Module, which is the base class for all neural networks in PyTorch.

In the constructor __init__, layers and activation functions are defined.

The first layer self.n1=nn.Linear(1,12) takes one input feature and maps it to 12 neurons.

The second layer self.n2=nn.Linear(12,14) processes the 12 outputs and maps them to 14 neurons.

The third layer self.n3=nn.Linear(14,1) reduces the 14 features back to a single output.

The activation function self.relu=nn.ReLU() introduces non-linearity, helping the network learn complex patterns.

A history dictionary is initialized to store the loss values during training for performance tracking.

The forward function defines how input data flows through the network layers.

Input x is first passed through n1 and activated by ReLU, then through n2 with ReLU again.

Finally, the processed data passes through n3 to produce the output, which is returned.


## Neural Network Model :

<img width="888" height="794" alt="image" src="https://github.com/user-attachments/assets/137f8995-27ca-4f5a-920c-094c30ac7b4a" />



## DESIGN STEPS :

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:A.Lahari
### Register Number:212223230111
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,12)
        self.n2=nn.Linear(12,14)
        self.n3=nn.Linear(14,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x

Lahari_brain=Neuralnet()
criteria=nn.MSELoss()
optimizer=optim.RMSprop(Lahari_brain.parameters(),lr=0.001)

def train_model(Lahari_brain,x_train,y_train,criteria,optmizer,epochs=4000):
    for i in range(epochs):
        optimizer.zero_grad()
        loss=criteria(Lahari_brain(x_train),y_train)
        loss.backward()
        optimizer.step()

        Lahari_brain.history['loss'].append(loss.item())
        if i%200==0:
            print(f"Epoch [{i}/epochs], loss: {loss.item():.6f}")


```
## Dataset Information:

<img width="188" height="525" alt="image" src="https://github.com/user-attachments/assets/b1d87418-6720-433e-ab35-d3a29a14c7f1" />


## OUTPUT:

### Training Loss Vs Iteration Plot

<img width="732" height="588" alt="image" src="https://github.com/user-attachments/assets/2882c2b7-cad8-4a09-b0cd-3745b2b84d98" />


### New Sample Data Prediction

<img width="901" height="124" alt="image" src="https://github.com/user-attachments/assets/bf02bd42-d788-4bd3-98b1-139e47057d5c" />


## RESULT :

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
