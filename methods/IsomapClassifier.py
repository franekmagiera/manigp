'''
Copyright: 2019-present Patryk Orzechowski
Licence: MIT
'''

from sklearn.base import BaseEstimator
from sklearn.manifold import Isomap
from sklearn.metrics import confusion_matrix
from utils.munkres import Munkres, make_cost_matrix
from sklearn.cluster import KMeans



class IsomapClassifier(BaseEstimator):

  def __init__(self,n_neighbors=5, n_components=2, n_clusters=2, eigen_solver='auto',random_state=3319):
    self.n_neighbors=n_neighbors
    self.n_components=n_components
    self.n_clusters=n_clusters
    self.random_state=random_state


  def fit(self,X,y):
    #creating a manifold on training data
    self.model=Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components, eigen_solver=self.eigen_solver).fit(X,y)
    #determining centroids for given classes
    self.centroids = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.model.transform(X))
    labels = self.centroids.predict(self.model.transform(X)) # Every point is assigned to a certain cluster.
    #assigning each centroid to the correct cluster
    confusion_m = confusion_matrix(y, labels)
    m = Munkres()
    cost_m = make_cost_matrix(confusion_m)
    target_cluster = m.compute(cost_m) # (target, cluster) assignment pairs.
    #saving mapping for predictions
    self.mapping = {cluster : target for target, cluster in dict(target_cluster).items()}


  def predict(self,X_test):
    #transforming test set using manifold learning method
    X_trans=self.model.transform(X_test)
    #assigning each of the points to the closest centroid
    labels = self.centroids.predict(X_trans)
    y_pred = list(map(self.mapping.get, labels))
    return y_pred


hyper_params = {
    'n_neighbors': [5,6,7,8,9,10,15,20],
    'eigen_solver' : ['arpack','dense'],
}


est=IsomapClassifier()