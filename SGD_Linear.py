import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    
def compute_gradient(x,y,w,m):
    grad=(-2/m)*((y-w.dot(x.T)).dot(x))
    return(grad)
    

def compute_loss(y_hat,y_true):
    err = (y_hat - y_true)**2
    return(err)
    
    
def stochastic_gradient_descent(df,lrate):
    m=len(df)
    initial_w=np.zeros(df.shape[1])
    final_w=np.zeros(df.shape[1])
    #b=0
    loss=[]
    weights=[]
    iteration=[]
    new_w=initial_w
    total_loss=0
    plotting_interval=[x*100 for x in range(1,1000)]
    for k in range(1,100000):
        s=df.sample(1)
        s['bias']=1
        x=s.iloc[:,1:].values
        y=s.iloc[:,0].values
        new_w = initial_w - (lrate * compute_gradient(x,y,initial_w,m))
        y_hat = new_w.dot(x.T)
        total_loss += compute_loss(y_hat,y)
        if k in plotting_interval:
            iteration.append(k)
            loss.append(total_loss/k)
            weights.append(initial_w)
        initial_w=new_w
    compute_norm(iteration,weights)
    plot_sgd(iteration,loss)
    #print(new_w)
    bias=new_w[len(new_w)-1:]
    #print(new_w[:len(new_w)-1])
    return(new_w[:len(new_w)-1], bias)
    #return(new_w)

    
def test_development_data(df_dev,w,b):
    x_dev=df_dev.iloc[:,1:].values
    y_dev=df_dev.iloc[:,0].values
    dev_error_count=0
    size=1
    for n in range(len(y_dev)):
        y_pred = np.sign(w.dot(x_dev[n].T) + b)
        if y_dev[n] != y_pred:
            dev_error_count += 1
        size += 1
    dev_error_rate = dev_error_count/float(size)
    dev_accuracy = 1 - dev_error_rate
    return(dev_accuracy)
    
            
def test_sgd(x_test,y_test,w,bias,lr):
    error_count=0
    test_iterations=[]
    test_accuracy=[]
    size=1
    #k_values=[5,10,15,20,25,30,35,40]
    for n in range(len(y_test)):
        y_pred = np.sign(w.dot(x_test[n].T) + bias)
        #print(y_pred)
        if y_test[n]!=y_pred:
            error_count += 1
        #if n in k_values:
        error_rate= 1-(error_count/float(size))
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
        
def plot_sgd(k,error):
    plt.style.use("ggplot")
    plt.figure(figsize=(10,5))
    plt.plot(k,error, color='purple')
    plt.xlabel("Iteration")
    plt.ylabel("Average Squared Error")
    plt.title("Average Squared Error as a function of Iteration")
    plt.show()
    

def compute_norm(k, weights):
    norm=[]
    for i in weights:
        norm.append(np.linalg.norm(i[1:len(i)-1]))
    plt.figure(figsize=(10,5))
    plt.style.use("ggplot")
    plt.plot(k,norm, color='green')
    plt.xlabel("Iterations")
    plt.ylabel("L2 Norm of weight vectors")
    plt.title("L2 norm of weight vectors as a function of Iterations")
    plt.show()
    
    

def main():
    final_weight=[]
    final_bias=[]
    df=pd.read_csv("A3.train.csv")
    #df=df.sample(frac=.1)
    split = np.random.rand(len(df)) < 0.91
    df_train = df[split]
    df_dev = df[~split]
    df2=pd.read_csv("A3.test.csv")
    #df2['bias']=1
    x_test=df2.iloc[:,1:].values
    y_test=df2.iloc[:,0].values
    lrate= [0.8, 0.001, 0.00001]
    for lr in lrate:
        w,b=stochastic_gradient_descent(df_train,lr)
        final_weight.append(w)
        final_bias.append(b)
    #print("Running Linear Regression on Test Data:")
    for i in range(len(final_weight)):
        print("Weight Vector:", final_weight[i], " and Bias: ", final_bias[i])
        print("Development accuracy for model: ", test_development_data(df_dev,final_weight[i],final_bias[i]))
        test_sgd(x_test,y_test, final_weight[i],final_bias[i],lrate[i]) 
        
if __name__ == '__main__':
    main()
    
    
