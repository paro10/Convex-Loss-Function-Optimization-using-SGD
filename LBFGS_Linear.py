import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sp
#import fmin_bfgs
import sys

def compute_loss(y_hat,y_true):
    err = np.sum((y_hat - y_true)**2)
    return(err)

def compute_cost(x, y, w): 
    m = len(y)
    cost = np.sum((w.dot(x.T)-y)**2)/(2*m)
    return(cost)

def compute_gradient(x, y, w, alpha, iterations):
    cost_list = []
    m = len(y)
    #for i in range(iterations):
        #activation = w.dot(x.T)
        #loss = y - activation
        #gradient = (-2/m)*((loss).dot(x))
    gradient=(-2/m)*((y-w.dot(x.T)).dot(x))
    w = w - alpha*gradient
        #cost = cost_function(x, y, w)
        #cost_list.append(cost)
    return(w)

def f(w):
    return(compute_cost(x,y,w))
            
def fprime(w):
    return(compute_gradient(x, y, w,0.001,500))


def test_gradient_descent(x_test,y_test,w,b):
    error_count=0
    test_iterations=[]
    test_accuracy=[]
    size=1
    y_hat = np.sign(w.dot(x_test.T) + b)
    for i in range(len(y_test)):
        if y_test[i] != y_hat[i]:
            error_count += 1
        test_accuracy.append(1 - (error_count/float(size)))
        test_iterations.append(i)
        size += 1
    plt.figure(figsize=(15,5))
    plt.style.use('ggplot')
    plt.title("Accuracy as a function of Iterations")
    plt.xlabel("No. of Iterations")
    plt.ylabel("Test set accuracy")
    plt.plot(test_iterations,test_accuracy, color="purple")
    plt.show()
    print("Accuracy of Test set:", (1-(error_count/float(size))))


def main():
    df=pd.read_csv("A3.train.csv")
    df['bias'] = 1
    x=df.iloc[:,1:].values
    y=df.iloc[:,0].values
    w=np.zeros(x.shape[1])
    alpha=0.001
    #p=fmin_bfgs(f, w, fprime, disp=True, maxiter=400)
    print(sp.minimize(f,w,method="BFGS"))
    final_weight = np.array([1.01605809,  2.88433943, -0.36064842, -0.22038431, -0.19281313,
        1.52707519,  0.88530152, -0.02542996])
    bias= -2.87098163
    #print(final_weight)
    df_test=pd.read_csv("A3.test.csv")
    x_test=df_test.iloc[:,1:].values
    y_test=df_test.iloc[:,0].values
    test_gradient_descent(x_test,y_test,final_weight,bias)
        
    
        
if __name__ == '__main__':
    main()
    
