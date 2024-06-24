import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class BrownianMotion(object):
    def __init__(self,variance,avg,steps,dtime):
        self.__variance=variance
        self.__steps=steps
        self.__dtime=dtime
        self.__avg=avg
    
    def get_stocks(self):
        s=np.array([100*np.ones(self.__steps) for i in range(len(self.__variance))])
        for k in range(len(self.__variance)):
          for i in range(self.__steps-1):
            m=self.__avg-np.sqrt(self.__dtime)**2/2
            s[k][i+1]=s[k][i]+(m*self.__dtime+self.__variance[k]*np.random.normal(loc=0,scale=1))
  
        return s 
    def plot(self):
       y=self.get_stocks()
       cmap = mpl.colormaps['plasma']
       colors = cmap(np.linspace(0, 1, len(self.__variance)))
       colormap = np.array([colors[i] for i in range(len(colors))])
       for i in range(len(self.__variance)): 
        plt.scatter(range(self.__steps),y[i],c=colormap[i],label='s=%s'%(self.__variance[i]))
       plt.legend()    
       plt.title("Brownian Motion with different variance")
       plt.xlabel("Time")
       plt.ylabel("Asset Value")
       plt.show()


mu = 1
n = 1000
sigma = np.linspace(0.5, 4.5, 9)
dt = 0.001
BrownianMotion(sigma,mu,n,dt).plot()
