import numpy as np
trX=np.linspace(-1,1,101) #Creating our training data. Its a feature of a hypothetical data-set. creates a vector from -1 to 1 with 101 samples. evenly spaced/sampled.

trY=2*trX+np.random.randn(*trX.shape)*0.33 #output values of the data-set that we will predict with linear regression. 
X=T.scalar() #defining two symbolic variables of scalar types that will be used to pass on values from the tuples.
Y=T.scalar() 
def model(X,w): #our custom linear regression function with no bias (intercept). We actually need to learn the weight "w" here that we will do so by gradient descent. 
    return X*w
w=theano.shared(np.asarray(0., dtype=theano.config.floatX)) #Defining the parameter of our model. Its of hybrid "shared" variable type which is a type of theano datatype used when value needs to be shared with functions. 

#print theano.config.floatX
y=model(X,w) 
cost=T.mean(T.sqr(y-Y)) #defining our cost function i.e. mean square error. 
gradient=T.grad(cost=cost,wrt=w) #defining gradient descent here i.e. we want to minimize cost function w.r.t our model's parameter w.
updates=[[w,w-gradient*0.01]] #defining the update step of gradient descent iteration where learning rate = 0.01.
train=theano.function(inputs=[X,Y],outputs=cost,updates=updates,allow_input_downcast=True) #compiling into a theano function so that it could be called later and values may be passed.

for i in range(100):
    for x,y in zip(trX,trY): #zip forms tuples. we are iterating over our records/instances and getting values from our (predictor,target) tuple to pass to our model to be trained.
        train(x,y) #passing the training data to our model to be trained.
