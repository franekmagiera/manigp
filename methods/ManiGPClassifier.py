'''
Copyright: 2019-present Patryk Orzechowski
Licence: GNU GPLv3

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''


from deap import base, creator, gp, tools
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import operator as op
import utils.operators as ops # From hibachi.
from utils.metrics import balanced_accuracy_score
import pandas as pd
import random
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.munkres import Munkres, make_cost_matrix



# Copied from hibachi.
def create_pset(in_type, in_types_length, out_type):
  pset = gp.PrimitiveSetTyped("MAIN",
                             itertools.repeat(in_type, in_types_length),
                             out_type,
                             prefix="x")
  # basic operators 
  pset.addPrimitive(ops.addition, [float,float], float)
  pset.addPrimitive(ops.subtract, [float,float], float)
  pset.addPrimitive(ops.multiply, [float,float], float)
  pset.addPrimitive(ops.safediv, [float,float], float)
  pset.addPrimitive(ops.modulus, [float,float], float)
  pset.addPrimitive(ops.plus_mod_two, [float,float], float)
  # logic operators
  pset.addPrimitive(ops.equal, [float, float], float)
  pset.addPrimitive(ops.not_equal, [float, float], float)
  pset.addPrimitive(ops.gt, [float, float], float)
  pset.addPrimitive(ops.lt, [float, float], float)
  pset.addPrimitive(ops.AND, [float, float], float)
  pset.addPrimitive(ops.OR, [float, float], float)
  pset.addPrimitive(ops.xor, [float,float], float)
  # bitwise operators 
  pset.addPrimitive(ops.bitand, [float,float], float)
  pset.addPrimitive(ops.bitor, [float,float], float)
  pset.addPrimitive(ops.bitxor, [float,float], float)
  # unary operators 
  pset.addPrimitive(op.abs, [float], float)
  pset.addPrimitive(ops.NOT, [float], float)
  pset.addPrimitive(ops.factorial, [float], float)
  # large operators 
  pset.addPrimitive(ops.power, [float,float], float)
  pset.addPrimitive(ops.logAofB, [float,float], float)
  pset.addPrimitive(ops.permute, [float,float], float)
  pset.addPrimitive(ops.choose, [float,float], float)
  # misc operators 
  pset.addPrimitive(ops.left, [float,float], float)
  pset.addPrimitive(ops.right, [float,float], float)
  pset.addPrimitive(min, [float,float], float)
  pset.addPrimitive(max, [float,float], float)
  # terminals 
  randval = "rand" + str(random.random())[2:]  # so it can rerun from ipython
  pset.addEphemeralConstant(randval,
                            lambda: random.random() * 100,
                            float)
  pset.addTerminal(0.0, float)
  pset.addTerminal(1.0, float)
  return pset

def create_toolbox(weights, pset, min_tree_height, max_tree_height, n_components):
  creator.create("FitnessFunction", base.Fitness, weights=weights)
  creator.create("Tree", gp.PrimitiveTree, pset=pset)
  creator.create("Individual", list, fitness=creator.FitnessFunction)

  toolbox = base.Toolbox()
  toolbox.register("expr",
                  gp.genHalfAndHalf,
                  pset=pset,
                  min_=min_tree_height,
                  max_=max_tree_height)
  toolbox.register("tree",
                  tools.initIterate,
                  creator.Tree,
                  toolbox.expr)
  toolbox.register("individual",
                  tools.initRepeat,
                  creator.Individual,
                  toolbox.tree,
                  n=n_components)
  toolbox.register("compile",
                  gp.compile,
                  pset=pset)
  toolbox.register("population",
                  tools.initRepeat,
                  list,
                  toolbox.individual)
  toolbox.register("selectBest", tools.selBest)
#  toolbox.register("selectTournament", tools.selTournament, tournsize=7)
  toolbox.register("mate", gp.cxOnePoint)
  toolbox.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=2)
  toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # Returns a tuple of one tree.
  toolbox.register("shrink", gp.mutShrink) # Returns a tuple of one tree.
#  toolbox.register("evaluate", evaluate, X_=X_train, y=y_train, random_state=random_state)
  return toolbox



# Tree from string with a given Primitive Set of operations.
tfs = lambda string, pset : gp.PrimitiveTree.from_string(string, pset)

# Get hight of a tree.
get_height = lambda tree : op.attrgetter('height')(tree)

def check_in_which_slice(angle, n, slices):
    for i in range(n):
        if slices[i][0] < angle < slices[i][1]:
            return i
    # Points on the borders are assigned a different label in order to handicap balanced accuracy score.
    return n


class ManiGPClassifier(BaseEstimator):


  def evaluate(self,individual, X, y, random_state):
    X_new = self.reduce(individual, X)

    if self.fitness_function == "kmeans":
      # Clustering of the reduced dataset.
      centroids = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X_new)
      labels = centroids.labels_ # Every point is assigned to a certain cluster.
      confusion_m = confusion_matrix(y, labels)
      m = Munkres()
      cost_m = make_cost_matrix(confusion_m)
      target_cluster = m.compute(cost_m) # (target, cluster) assignment pairs.
      cluster_target = {cluster : target for target, cluster in dict(target_cluster).items()}
      y_pred = list(map(cluster_target.get, labels))    
      return balanced_accuracy_score(y, y_pred)
    elif self.fitness_function == "nn":
      # n_clusters is equal to the number of classes.
      # n_neighbors is always odd and bigger than number of classes. This way classification is unambiguous.
      n_neighbors = self.n_clusters + (1 if self.n_clusters % 2 == 0 else 2)
      # n_neighbors + 1 because the class of the point itself is not taken into account.
      neighbors = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_new)
      nearest_neighbors = neighbors.kneighbors(X_new, return_distance=False)[:,1:]
      classes = y[nearest_neighbors]
      y_pred = mode(classes, axis=1)[0].reshape(len(y),)
      return balanced_accuracy_score(y, y_pred)
    elif self.fitness_function == "angles":
      angles = np.apply_along_axis(lambda row: math.atan2(row[1], row[0]), axis=1, arr=X_new)
      # Mapping from (-pi,pi) to (0, 2*pi)
      angles = (2*np.pi + angles) * (angles < 0) + angles*(angles > 0)
      y_pred = list(map(lambda angle: check_in_which_slice(angle, self.n_clusters, self.slices), angles))
      return balanced_accuracy_score(y, y_pred)




  def reduce(self, individual, X):
    n_components = len(individual)
    m, n = X.shape[0], X.shape[1]
    X_new = np.zeros((m, n_components), dtype='float64')

    # Dimensionality reduction.
    for i in range(n_components): # For every tree/variable in reduced dataset.
      f = self.toolbox.compile(individual[i])
      for j in range(m): # For every row. Every row is passed through a tree.
        X_new[j, i] = f(*X.iloc[j]) if isinstance(X, pd.DataFrame) else f(*X[j,:]) # X can be a pandas DataFrame or numpy array.
    return X_new


#  def __init__(self, mutpb=0.9, cxpb=0.1, pop_size=100, n_iter=500, tourn_size=7, weights = (1.0,),min_tree_height = 1, max_tree_height = 5, n_components = 2, random_state=3319):
  def __init__(self, mutpb=0.9, cxpb=0.1, pop_size=100, n_iter=500, tourn_size=7, weights = (1.0,),min_tree_height = 1, max_tree_height = 4, n_components = 2, random_state=3319, fitness_function="kmeans", predictor="kmeans"):
    self.mutpb=mutpb
    self.cxpb=cxpb
    self.pop_size=pop_size
    self.n_iter=n_iter
    self.tourn_size=tourn_size
    self.weights=weights
    self.min_tree_height = min_tree_height
    self.max_tree_height = max_tree_height
    self.n_components = n_components
    self.random_state=random_state
    self.fitness_function = fitness_function
    self.predictor = predictor
    self.rejected = 0 # Number of rejected trees.
    self.cx_count = 0 # Numer of crossovers.
    self.mut_count = 0 # Number of mutations.
    random.seed(random_state)

  def fit(self,X,y):
    pset = create_pset(in_type=float, in_types_length=X.shape[1], out_type=float)
    self.toolbox = create_toolbox(weights=self.weights,
                                pset=pset,
                                min_tree_height=self.min_tree_height,
                                max_tree_height=self.max_tree_height,
                                n_components=self.n_components)

    self.toolbox.register("evaluate", self.evaluate, X=X, y=y, random_state=self.random_state)

    population = self.toolbox.population(self.pop_size)
    best_individuals = []
    self.n_clusters=len(Counter(y))
    self.slices = [(i*2*math.pi/self.n_clusters, (i+1)*2*math.pi/self.n_clusters) for i in range(self.n_clusters)]
    self.rejected = 0
    self.cx_count = 0
    self.mut_count = 0

    for g in range(self.n_iter):
      population = self.toolbox.selectBest(population, self.pop_size)
      best_individuals.append(self.toolbox.selectBest(population, 1)[0])
      random.shuffle(population)
      for parent1, parent2 in zip(population[::2], population[1::2]):
        if random.random() < self.cxpb:
          self.cx_count += 1
          child1 = self.toolbox.clone(parent1)
          child2 = self.toolbox.clone(parent2)
          for i in range(self.n_components):
            self.toolbox.mate(child1[i], child2[i])
          reject = False
          for i in range(self.n_components):
            if get_height(child1[i]) > self.max_tree_height:
              reject = True
              self.rejected += 1
              break
          if not reject:
            del child1.fitness.values
            population.append(child1)
          reject = False
          for i in range(self.n_components):
            if get_height(child2[i]) > self.max_tree_height:
              reject = True
              self.rejected += 1
              break
          if not reject:
            del child2.fitness.values
            population.append(child2)

      for individual in population.copy():
        if random.random() < self.mutpb:
          self.mut_count += 1
          mutant = self.toolbox.clone(individual)
          for i in range(self.n_components):
            self.toolbox.mutate(mutant[i])
          reject = False
          for i in range(self.n_components):
            if get_height(mutant[i]) > self.max_tree_height:
              reject = True
              self.rejected += 1
              break
          if not reject:
            del mutant.fitness.values
            population.append(mutant)

      invalid_ind = [ind for ind in population if not ind.fitness.valid]
      fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit,)

#      population = self.toolbox.selectTournament(population, self.tourn_size)

    best_individuals.append(self.toolbox.selectBest(population, 1)[0])
    self.best_fitness = best_individuals[-1].fitness.values[0]
    self.model=best_individuals[-1]
    X_new = self.reduce(self.model, X)
    self.centroids=KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X_new)
    labels = self.centroids.predict(X_new)
    confusion_m = confusion_matrix(y, labels)
    m = Munkres()
    cost_m = make_cost_matrix(confusion_m)
    target_cluster = m.compute(cost_m) # (target, cluster) assignment pairs.
    self.mapping = {cluster : target for target, cluster in dict(target_cluster).items()}

    # Nearest neighbors.
    self.neighbors = NearestNeighbors(n_neighbors=1).fit(X_new)
    self.y_train = y
  def predict(self,X_test):
    #transforming test set using manifold learning method
    X_trans=self.reduce(self.model,X_test)

    if self.predictor == "kmeans":
      #assigning each of the points to the closest centroid
      labels = self.centroids.predict(X_trans)
      y_pred = list(map(self.mapping.get, labels))
      return y_pred
    elif self.predictor == "nn":
      indices = self.neighbors.kneighbors(X_trans, return_distance=False)
      y_pred = (self.y_train)[indices].reshape((len(X_trans),))
      return y_pred





est = ManiGPClassifier()

hyper_params=[
   {'cxpb':[0.1], 'mutpb':[0.9], 'fitness_function' : ['nn']},
   {'cxpb':[0.5], 'mutpb':[0.5], 'fitness_function' : ['nn']},
   {'cxpb':[0.9], 'mutpb':[0.1], 'fitness_function' : ['nn']},
   {'cxpb':[0.1], 'mutpb':[0.9], 'fitness_function' : ['angles']},
   {'cxpb':[0.5], 'mutpb':[0.5], 'fitness_function' : ['angles']},
   {'cxpb':[0.9], 'mutpb':[0.1], 'fitness_function' : ['angles']},
]

