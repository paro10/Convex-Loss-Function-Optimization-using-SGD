import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def compute_gradient(x,y,w):
    ac = y*(w.dot(x.T)) #activation function
    grad = (1/(1 + math.exp(ac)))*(-y*x)   #gradient
    return(grad)
    
    
def compute_loss(x,y,w):
    ac = y*(w.dot(x.T)) #activation function
    log_loss = math.log(1 + math.exp(-ac))
    return(log_loss)


def compute_squared_error(y_hat,y):
    error = (y_hat - y)**2 #squared error
    return(error)

def test_development_data(df_dev,w):
    x_dev=df_dev.iloc[:,1:].values  #converting dataframe to array
    y_dev=df_dev.iloc[:,0].values   #converting dataframe to array
    dev_error_count=0
    size=1
    for n in range(len(y_dev)):
        y_pred = sigmoid_predict(x_dev[n],y_dev[n],w)    #passing data and parameter to sigmoid function
        if y_dev[n] != y_pred:
            dev_error_count += 1    # count errors if predicted value != true value
        size += 1
    dev_error_rate = dev_error_count/float(size)
    dev_accuracy = 1 - dev_error_rate   #accuracy calculation
    return(dev_accuracy)
    
    
def test_sgd(df2,w):
    x_test=df2.iloc[:,1:].values    #converting dataframe to array
    y_test=df2.iloc[:,0].values    #converting dataframe to array
    error_count=0
    test_iterations=[]
    test_accuracy=[]
    size=1
    #k_values=[5,10,15,20,25,30,35,40]
    for n in range(len(y_test)):
        y_pred = sigmoid_predict(x_test[n],y_test[n],w)    #sigmoid maps to -1 or +1
        #print(y_pred)
        if y_test[n]!=y_pred:
            error_count += 1   #count errors if predicted value not equal to actual label
        #if n in k_values:
        error_rate= 1-(error_count/float(size))  #accuracy calculation 
        #print("err rate",error_rate)
        test_accuracy.append(error_rate*100)  
        test_iterations.append(size)
        size += 1
    #print(test_accuracy)
    plt.style.use("ggplot")
    plt.figure(figsize=(15,5))
    plt.plot(test_iterations,test_accuracy)
    plt.title("Test Set Accuracy as a function of Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Test Set accuracy")
    plt.show() 
    print("Test set accuracy: ",1-(error_count/float(n)))

    
def sigmoid_predict(x,y,w):
    ac = y*(w.dot(x.T))
    sigmoid_output = 1/(1 + math.exp(ac))    
    #print(sigmoid_output)
    y_hat = 1 if sigmoid_output > 0.5 else -1   #binary classification based on threshold of 0.5
    #print('yhat',y_hat)
    return(y_hat)
    
def stochastic_gradient_descent(df,lrate):
    m=len(df)
    w=np.zeros(df.shape[1]-1)
    #final_w=np.zeros(df.shape[1])
    loss=[]
    weights=[]
    iteration=[]
    #new_w=initial_w
    total_error=0
    final_cost=0
    plotting_interval=[x*100 for x in range(0,1000)]
    for k in range(1,100000):
        s=df.sample(1)
        x=s.iloc[:,1:].values
        y=s.iloc[:,0].values
        #print("y",y)
        y_hat = sigmoid_predict(x,y,w)
        initial_cost = compute_loss(x,y,w)
        if initial_cost - final_cost  >= 0.0000001:     #update weight vectors if not converged
            w = w - (lrate * compute_gradient(x,y,w))
        final_cost = compute_loss(x,y,w)
        total_error += compute_squared_error(y_hat,y)
        if k in plotting_interval:
            iteration.append(k)
            loss.append(total_error/float(k))
            weights.append(w)
        #initial_w=new_w
    compute_norm(iteration,weights)
    plot_sgd(iteration,loss)
    #print(new_w)
    #bias=new_w[len(new_w)-1:]
    #print(new_w[:len(new_w)-1])
    #return(new_w[:len(new_w)-1], bias)    
    return(w)


def compute_norm(k, weights):
    norm=[]
    for i in weights:
        w_without_bias = np.delete(i,8)   # removed bias from weight vector
        norm.append(np.linalg.norm(w_without_bias)) #calculate norm without bias
    plt.figure(figsize=(15,5))
    plt.style.use("ggplot")
    plt.plot(k,norm, color='green')
    plt.xlabel("Iterations")
    plt.ylabel("L2 Norm of weight vectors")
    plt.title("L2 norm of weight vectors as a function of Iterations")
    plt.show()
    
    
    
def plot_sgd(k,loss):
    plt.style.use("ggplot")
    plt.figure(figsize=(15,5))
    plt.plot(k,loss, color='purple')
    plt.xlabel("Iteration")
    plt.ylabel("Average Squared Error")
    plt.title("Average Squared Error as a function of Iteration")
    plt.show()



def main():
    final_weight=[]
    final_bias=[]
    df=pd.read_csv("A3.train.csv")
    df['bias']=1
    split = np.random.rand(len(df)) < 0.91
    df_train = df[split]
    df_dev = df[~split]
    df2=pd.read_csv("A3.test.csv")
    df2['bias']=1
    #x_test=df2.iloc[:,1:].values
    #y_test=df2.iloc[:,0].values
    lrate= [0.8, 0.001, 0.00001]
    for lr in lrate:
        w=stochastic_gradient_descent(df_train, lr)
        #b=w[len(w)-1:]
        final_weight.append(w)
        #final_bias.append(b)
    #print("Running Linear Regression on Test Data:")
    for i in range(len(final_weight)):
        print("Weight Vector:", final_weight[i])
        print("Development accuracy for model: ", test_development_data(df_dev,final_weight[i]))
        test_sgd(df2, final_weight[i]) 
    
if __name__ == '__main__':
    main()