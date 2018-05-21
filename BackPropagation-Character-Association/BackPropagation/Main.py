'''
Created on May 21, 2018

@author: Parsa
'''
import BackPropagation as b
def main():
    x = b.BackPropagation(activationFunction="bipolar_sigmoid", alpha=0.1, row=9, col=7, inputLength=63, targetLength=63, hiddenUnit=13, numOfLearningInputs=7)
    x.training()
    x.info()
    x.test()
if __name__ == '__main__':
    main()
    