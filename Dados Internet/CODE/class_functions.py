import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances as pdist
from sklearn.decomposition import PCA


### CCA

class CCA:
    def __init__(self, p, lmbd, alpha):
        """
        Creates the CCA object. 
        Parameters
        ----------
        p : int
            The new dimension.
        lmbd : float
            Distance limit to update points. It decreases over time : lambda(t) = lambda/(t+1).
        alpha : float
            Learning rate. It decreases over time : alpha(t) = alpha/(t+1)
        """
        self.p = p
        self.lmbd = lmbd
        self.alpha = alpha    

    def _stress(self, dist_y, dist_x, lmbda):
        """
        Calculates the stress function given the distances in original space (dist_y)
        and the distances in reduced space (dist_x).
        Parameters
        ----------
        dist_y : numpy.array
            Array with distances in original space.
        dist_x : numpy.array
            Array with distances in reduced space.
        lmbda : float
            Distance limit to update points.
        """
        stress = np.mean((dist_y - dist_x)**2 * (lmbda > dist_x).astype(int))
        return stress

    def run(self, data_y, q_max=10, show=False, tol=1e-6, random_state=None, init='PCA'):
        """
        Method to reduce dimension. Every iteration run all points. The new data
        is stored in attribute 'data_x'.
        Parameters
        ----------
        data_y : numpy.array
            Array with the original data.
        q_max : int (default = 10)
            Number of iterations. Each iteration run all points in 'data_y'.
        show : boolean (default = False)
            If True, shows the stress curve along time.
        tol : float (default = 1e-4)
            Tolerance for the stopping criteria.

        Returns
        -------
        data_x : numpy.array
            New data representation.
        """
        self.data_y = data_y
        n = len(data_y)
        triu = np.triu_indices(n, 1)
        dist_y = pdist(data_y)
        if init == 'PCA':
            data_x = PCA(self.p).fit_transform(data_y)
        elif init == 'Random':
            np.random.seed(random_state)
            new_shape = (data_y.shape[0], self.p)
            data_x = np.random.random(new_shape)
        stress = np.zeros(q_max)
        print("Progress: 0.0%", end='\r')
        for q in range(q_max):
            alpha = max(0.001, self.alpha/(1+q))
            lmbda = max(0.1, self.lmbd/(1+q))
            for i in range(n):
                dist_x = cdist(data_x[i].reshape(1,-1), data_x)    
                dy = np.delete(dist_y[i],i,0) 
                dx = np.delete(dist_x,i,1)
                delta_x = (alpha*(lmbda > dx)*(dy - dx)/dx).reshape((-1,1))*(data_x[i] - np.delete(data_x, i, 0))
                delta_x = np.insert(delta_x, i, 0, axis=0)
                data_x -= delta_x
            dist_x = pdist(data_x)
            stress[q] = self._stress(dist_y[triu], dist_x[triu], lmbda)
            if stress[q] < tol:
                print("Progress: 100.00%")
                print(f"Tol achieved in iteration {q}")
                break
            print(f"Progress: {round((q+1)*100/q_max,2)}%  ", end='\r')
        if show:
            plt.plot(np.arange(q_max), stress, marker='.', c='black')
            plt.xlabel("Iteration")
            plt.ylabel("Stress")
            plt.show()
        print()
        self.data_x = data_x
        return data_x

    def plotYX(self):
        """
        Creates the dy dx Representation with the original and the reduced data.
        """
        reduced_data = self.data_x
        original_data = self.data_y
        dy = []
        dx = []

        for i in range(reduced_data.shape[0]):
            y1 = reduced_data[i,:]
            x1 = original_data[i,:]
            for j in range(i+1, reduced_data.shape[0]):
                y2 = reduced_data[j,:]
                x2 = original_data[j,:]
                dy.append(np.linalg.norm(y2-y1))
                dx.append(np.linalg.norm(x2-x1))
        plt.scatter(dy,dx, c='black', s=1)
        lims = [
            np.min([plt.xlim(), plt.ylim()]),  # min of both axes
            np.max([plt.xlim(), plt.ylim()]),  # max of both axes
        ]
        plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.xlim(lims)
        plt.ylim(lims)
        plt.ylabel("Distance between points in original space")
        plt.xlabel("Distance between points in reduced space")
        plt.show()


### SOM

class SOM:
    def __init__(self, shape, m, sigma_0):
        self.shape = shape
        self.m = m
        self.n = 0
        self.sigma_0 = sigma_0
        self.t1 = 1000/np.log(self.sigma_0)
        self.map = np.array(list(map(lambda i: np.random.random(m)/10, np.arange(shape[0]*shape[1])))).reshape((shape[0], shape[1], m))
        self.jv, self.iv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
    def getN(self, n):
        return max(0.1*np.exp(-n/1000), 0.001)
    
    def getSigma(self, n):
        return max(self.sigma_0*np.exp(-n/self.t1), 0.001)
        
    def run(self, sample):
        l,c = self.shape
        centers = self.map.reshape(-1,self.m)
        argmin = np.argmin( np.linalg.norm(centers-sample, axis=1))
        i, j = argmin//c, argmin%c
        dij = ((self.iv - i)**2 + (self.jv-j)**2)**0.5
        h = np.exp(-dij**2/(2*self.getSigma(self.n)**2))
        centers = centers + (h*self.getN(self.n)).reshape((-1,1))*(sample - centers)
        self.map = centers.reshape((l,c,self.m))
        self.n += 1
        
    def activate(self, sample, return_dist=False):
        _,c = self.shape
        centers = self.map.reshape(-1,self.m)
        dists = np.linalg.norm(centers-sample, axis=1)
        argmin = np.argmin( dists )
        if return_dist:
            return (argmin//c, argmin%c), dists[argmin]
        return (argmin//c, argmin%c)
    
    def showMap(self, i, var_std=1, var_mean=0, cmap='inferno', return_map=False, show=True):
        c_reshape = self.map
        image_s = c_reshape[:,:,i]*var_std + var_mean
        if show:
            plt.imshow(image_s, cmap=cmap)
            plt.colorbar()
            plt.show()
        if return_map:
            return image_s
        
    def showUMatrix(self, cmap='inferno'):
        shape_u = (self.shape[0]*2-1, self.shape[1]*2-1)
        matrix_u = np.zeros(shape_u)
        l_u, c_u = matrix_u.shape
        
        for i in range(l_u):
            for j in range(c_u):
                if i%2 == 0:
                    if (i+j)%2 == 0:
                        continue
                    else:
                        n1 = som.map[i//2,(j-1)//2]
                        n2 = som.map[i//2,(j+1)//2]
                        matrix_u[i,j] = np.linalg.norm(n1-n2)
                else:
                    if (i+j)%2 == 0:
                        n1 = som.map[(i+1)//2,(j-1)//2]
                        n2 = som.map[(i-1)//2,(j+1)//2]
                        matrix_u[i,j] = np.linalg.norm(n1-n2)
                    else:
                        n1 = som.map[(i-1)//2,j//2]
                        n2 = som.map[(i+1)//2,j//2]
                        matrix_u[i,j] = np.linalg.norm(n1-n2)

        plt.imshow(matrix_u, cmap=cmap)
        plt.colorbar()
        plt.show()
    
    def loadSOM(self, path):
        self.map = pd.read_parquet(path).values.reshape((self.shape[0], self.shape[1], self.m))

    def saveSOM(self, filename):
        data = self.map.reshape((-1, self.m))
        pd.DataFrame(data, columns=list(map(lambda s: f"Var_{s}", np.arange(self.m)))).to_parquet(filename)

    def plotYX(self):
        """
        Creates the dy dx Representation with the original and the reduced data.
        """
        reduced_data = np.concatenate([self.iv.reshape(-1,1), self.jv.reshape(-1,1)], axis=1)
        original_data = self.map.reshape((-1, self.m))
        dy = []
        dx = []

        for i in range(reduced_data.shape[0]):
            y1 = reduced_data[i,:]
            x1 = original_data[i,:]
            for j in range(i+1, reduced_data.shape[0]):
                y2 = reduced_data[j,:]
                x2 = original_data[j,:]
                dy.append(np.linalg.norm(y2-y1))
                dx.append(np.linalg.norm(x2-x1))
        plt.scatter(dy,dx, c='black', s=1)
        plt.title("dydx Representation - SOM")
        plt.ylabel("Distance between neurons in space")
        plt.xlabel("Distance between neurons in map")
        plt.show()


### SOM-DL

class SOMDL:
    def __init__(self, shape, m, sigma_0, u=0.1):
        self.shape = shape
        self.m = m
        self.n = 0
        self.u = u
        self.sigma_0 = sigma_0
        self.t1 = 1000/np.log(self.sigma_0)
        self.map = np.array(list(map(lambda i: np.random.random(m)/10, np.arange(shape[0]*shape[1])))).reshape((shape[0], shape[1], m))
        self.jv, self.iv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    def getSigma(self, n):
        return max(self.sigma_0*np.exp(-n/self.t1), 0.001)
        
    def run(self, sample):
        l,c = self.shape
        centers = self.map.reshape(-1,self.m)
        argmin = np.argmin( np.linalg.norm(centers-sample, axis=1))
        i, j = argmin//c, argmin%c
        dij = ((self.iv - i)**2 + (self.jv-j)**2)**0.5
        h = np.exp(-dij**2/(2*self.getSigma(self.n)**2))
        diff = (sample - centers)
        centers = centers + (h*self.u).reshape((-1,1))*np.sign(diff)*diff**2
        self.map = centers.reshape((l,c,self.m))
        self.n += 1
        
    def activate(self, sample, return_dist=False):
        _,c = self.shape
        centers = self.map.reshape(-1,self.m)
        dists = np.linalg.norm(centers-sample, axis=1)
        argmin = np.argmin( dists )
        if return_dist:
            return (argmin//c, argmin%c), dists[argmin]
        return (argmin//c, argmin%c)
    
    def showMap(self, i, var_std=1, var_mean=0, cmap='inferno', return_map=False, show=True):
        c_reshape = self.map
        image_s = c_reshape[:,:,i]*var_std + var_mean
        if show:
            plt.imshow(image_s, cmap=cmap)
            plt.colorbar()
            plt.show()
        if return_map:
            return image_s
        
    def showUMatrix(self, cmap='inferno'):
        shape_u = (self.shape[0]*2-1, self.shape[1]*2-1)
        matrix_u = np.zeros(shape_u)
        l_u, c_u = matrix_u.shape
        
        for i in range(l_u):
            for j in range(c_u):
                if i%2 == 0:
                    if (i+j)%2 == 0:
                        continue
                    else:
                        n1 = som.map[i//2,(j-1)//2]
                        n2 = som.map[i//2,(j+1)//2]
                        matrix_u[i,j] = np.linalg.norm(n1-n2)
                else:
                    if (i+j)%2 == 0:
                        n1 = som.map[(i+1)//2,(j-1)//2]
                        n2 = som.map[(i-1)//2,(j+1)//2]
                        matrix_u[i,j] = np.linalg.norm(n1-n2)
                    else:
                        n1 = som.map[(i-1)//2,j//2]
                        n2 = som.map[(i+1)//2,j//2]
                        matrix_u[i,j] = np.linalg.norm(n1-n2)

        plt.imshow(matrix_u, cmap=cmap)
        plt.colorbar()
        plt.show()
    
    def loadSOM(self, path):
        self.map = pd.read_parquet(path).values.reshape((self.shape[0], self.shape[1], self.m))

    def saveSOM(self, filename):
        data = self.map.reshape((-1, self.m))
        pd.DataFrame(data, columns=list(map(lambda s: f"Var_{s}", np.arange(self.m)))).to_parquet(filename)

    def plotYX(self):
        """
        Creates the dy dx Representation with the original and the reduced data.
        """
        reduced_data = np.concatenate([self.iv.reshape(-1,1), self.jv.reshape(-1,1)], axis=1)
        original_data = self.map.reshape((-1, self.m))
        dy = []
        dx = []

        for i in range(reduced_data.shape[0]):
            y1 = reduced_data[i,:]
            x1 = original_data[i,:]
            for j in range(i+1, reduced_data.shape[0]):
                y2 = reduced_data[j,:]
                x2 = original_data[j,:]
                dy.append(np.linalg.norm(y2-y1))
                dx.append(np.linalg.norm(x2-x1))
        plt.scatter(dy,dx, c='black', s=1)
        plt.title("dydx Representation - SOM")
        plt.ylabel("Distance between neurons in space")
        plt.xlabel("Distance between neurons in map")
        plt.show()