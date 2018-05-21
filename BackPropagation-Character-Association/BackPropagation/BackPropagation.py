import math
import numpy as np

class BackPropagation:
    """this is a backpropagation for character association which uses one hidden layer.
    this class is used for autoassociative recognition but can be used for pattern recognition 
    or heteroassociative recognition if changes are applied to the targetLength. hiddenUnit is
    the number of neurons on the hidden layer and row and col refer to the number of rows and 
    columns for each input recieved from input.txt(learning patterns). the character.txt is 
    the testing data that can be applied to the network with noise. alpha is the learning rate 
    and numOfLearningInputs refer to the number of characters in the input.txt. activation function
    should as well be either sigmoid or bipolar_sigmoid"""
    def __init__(self, activationFunction, alpha, row, col, inputLength, targetLength, hiddenUnit, numOfLearningInputs):
        self.activationFunction = activationFunction
        self.alpha = alpha
        self.row = row
        self.col = col
        self.inputLength = inputLength
        self.targetLength = targetLength
        self.hiddenUnit = hiddenUnit
        self.numOfLearningInputs = numOfLearningInputs
        self.sigmoidNumpy = np.frompyfunc(self.sigmoid, 1, 1)
        self.sigmoidDerivativeNumpy = np.frompyfunc(self.sigmoidDerivative, 1, 1)
        self.toOutputNumpy = np.frompyfunc(self.toOutput, 1, 1)
        self.signToNumNumpy = np.frompyfunc(self.signToNum, 1, 1)
        if self.activationFunction == "sigmoid":
            self.error = 0.01
        elif self.activationFunction == "bipolar_sigmoid":
            self.error = 0.01
    def training(self):
        self.gettingInput()
        self.weightInitializing()
        i = 0; j = 0; self.iterations = 0
        while True:
            self.iterations += 1
            self.feedForward(i)
            self.backPropagation(i)
            self.weightUpdating()
            i += 1
            i %= self.numOfLearningInputs
            j += 1
            if i == 0:
                if (abs(self.t-self.y)).all()<self.error:
                    print("converged to:\n",self.error," error")
                    break
    def weightInitializing(self):
        self.v = np.mat(2*(np.random.rand(self.inputLength+1, self.hiddenUnit))-1)
        self.w = np.mat(2*(np.random.rand(self.hiddenUnit+1, self.targetLength))-1)
        self.input = np.insert(self.input, 0, 1, axis=2)
    def feedForward(self,i):
        self.z_in = np.dot(self.input[i],self.v)
        self.z = self.sigmoidNumpy(self.z_in)
        self.z = np.insert(self.z, 0, 1)
        self.y_in = np.dot(self.z, self.w)
        self.y = self.sigmoidNumpy(self.y_in)
        self.t = self.input[i,0,1:self.targetLength+1]
    def backPropagation(self, i):
        self.deltaY = np.multiply(self.t - self.y, self.sigmoidDerivativeNumpy(self.y))
        self.deltaW = self.alpha * np.dot(self.z.T, self.deltaY)
        self.deltaZ = np.multiply(np.dot(self.deltaY, self.w.T), self.sigmoidDerivativeNumpy(self.z))
        self.deltaZ = self.deltaZ[0,1:self.hiddenUnit+1]
        self.deltaV = self.alpha * np.dot(self.input[i].T, self.deltaZ)
    def weightUpdating(self):
        self.w += np.array(self.deltaW, np.float)
        self.v += np.array(self.deltaV, np.float)
    def sigmoid(self, x):
        if self.activationFunction == "sigmoid":
            return (1 / (1 + np.exp(-x)))
        elif self.activationFunction=="bipolar_sigmoid":
            return (2 / (1 + np.exp(-x))) - 1
    def sigmoidDerivative(self, x):
        if self.activationFunction == "sigmoid":
            return x*(1-x)
        elif self.activationFunction == "bipolar_sigmoid":
            return  0.5 * (1 + x) * (1 - x)
    def gettingInput(self):
        self.input = np.loadtxt("input.txt", dtype=np.str, comments="//")
        self.input = self.signToNumNumpy(self.input)
        self.input = np.reshape(self.input,(int(len(self.input)/self.row),1,self.inputLength))
    def test(self):
        self.inputTest = np.loadtxt("character.txt", dtype=np.str, comments="//")
        print("testing input:\n",self.inputTest)
        self.inputTest = self.signToNumNumpy(self.inputTest)
        self.inputTest = np.reshape(self.inputTest,(1,self.inputLength))
        self.inputTest = np.insert(self.inputTest, 0, 1)
        self.test_z_in = np.dot(self.inputTest, self.v)
        self.test_z = self.sigmoidNumpy(self.test_z_in)
        self.test_z = np.insert(self.test_z, 0,1,1)
        self.test_y_in = np.dot(self.test_z, self.w)
        self.test_y = self.sigmoidNumpy(self.test_y_in)
        self.test_y = self.toOutputNumpy(self.test_y)
        self.test_y = self.test_y.reshape(1,self.row,self.col)
        print("result:\n",self.test_y)
    def toOutput(self,x):
        if self.activationFunction=="bipolar_sigmoid":
            if x>=0:
                return 1
            elif x<0:
                return 0
        elif self.activationFunction == "sigmoid":
            if x>=0.5:
                return 1
            elif x<0.5:
                return 0
    def signToNum(self,x):
        if x=='#':
            return 1
        elif x=='*':
            return -1
    def info(self):
        print("number of iterations to converge:\n", self.iterations)
        print("number of learning inputs:\n", self.numOfLearningInputs)
        print("input_length:\n", self.inputLength)
        print("target_length:\n", self.targetLength)
        print("number of neurons in the Hidden layer:\n", self.hiddenUnit)
        print("alpha\n", self.alpha)
        print("Activation Function:\n", self.activationFunction)

