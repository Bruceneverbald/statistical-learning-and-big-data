#!/usr/bin/env python
# coding: utf-8


# ### Bruce 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
mydata = pd.read_excel('/Users/spike/Desktop/420/regression_data.xlsx')
data = mydata[['x','y']].to_numpy()
type(data)
x = data[:,0]
y = data[:,1]


# 1.Plot the data and the true function f* on a single figure.

# In[4]:


ytrue = list(map(lambda x: (x-2)*(x-1)*x*(x+1)*(x+2),x))
plt.plot(x,y,'ro',label = 'data')
plt.plot(x,ytrue,'bo',label = 'true function')
plt.xlabel('x')
plt.ylabel('y_ture')
plt.title("ture function")
plt.legend()
plt.grid()
plt.show()


# 2.Estimate the risk of the true function f* from the whole data set

# In[134]:


print(np.mean((y - ytrue)**2))


# 3.Prediction error against model complexity.

# In[135]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=5,include_bias=True)


# In[136]:


from sklearn.linear_model import LinearRegression
Risktrain,Risktest = [],[]
for i in range(11):
    poly = PolynomialFeatures(degree=i,include_bias=True)
    poly_features = poly.fit_transform(x.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features[0:30,],y[0:30])
    ypre = poly_reg_model.predict(poly_features)
    sqer = (ypre - y)**2
    Risktrain.append(np.mean(sqer[0:30]))
    Risktest.append(np.mean(sqer[30:]))


# In[137]:


import matplotlib.pyplot as plt
plt.plot(range(0,11),Risktest,label ='Test')
plt.plot(range(0,11),Risktrain,label = 'Train')
plt.legend()
plt.xlabel('model complexity')
plt.ylabel('prediction error')
plt.title("estimated risks")
plt.grid()
plt.show()


# In[138]:


poly = PolynomialFeatures(degree=5,include_bias=True)
poly_features = poly.fit_transform(x.reshape(-1, 1))
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features[0:30,],y[0:30])
ypre = poly_reg_model.predict(poly_features)


# 4.Based on plot from task 3, model with degree=5 is the most appropriate, as test_error minimised and train_error become stable

# In[139]:



plt.plot(x[30:], y[30:],'ro',label = 'test')
plt.plot(x, ypre,'go',label = 'fitted data')
plt.plot(x[0:30], y[0:30],'bo',label = 'training')
plt.legend()
plt.title('DATA')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()


# In[140]:


E1,E2= [],[]
polyn = PolynomialFeatures(degree=5,include_bias=True)
polyn_features = polyn.fit_transform(x.reshape(-1, 1))
polyn_reg_model = LinearRegression()
for n in range(10,51):
    polyn_reg_model.fit(polyn_features[0:n,:],y[0:n])
    ypre = polyn_reg_model.predict(polyn_features)
    sqer = (ypre - y)**2
    E1.append(np.log(np.mean(sqer[0:n])))
    E2.append(np.log(np.mean(sqer[n:])))


# 5.prediction error against amount of training data.

# In[141]:



plt.plot(range(10,51),E2,'r-',label = 'testing error')
plt.plot(range(10,51),E1,'b-',label = 'training error')
plt.legend()
plt.xlabel('training datasize')
plt.ylabel('prediction error')
plt.title("simple model")
plt.grid()
plt.show()

