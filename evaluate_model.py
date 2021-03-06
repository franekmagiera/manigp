import sys
import itertools
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, train_test_split, ParameterGrid, cross_val_score)
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import warnings
from time import time
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from read_file import read_file
import pdb
import numpy as np
from methods import *
import os.path
import copy

def evaluate_model(dataset, save_file, random_state, est, hyper_params):

    est_name = type(est).__name__
    # load data
    X, y, feature_names = read_file(dataset)
    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.75,
                                                        test_size=0.25,
                                                        stratify=y,
                                                        random_state=None)
    assert(random_state!=None)
    # scale and normalize the data
    dataname = dataset.split('/')[-1][:-7]
    
    # Grid Search
    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
        # grid_est.fit(X_train,y_train)
        param_grid = list(ParameterGrid(hyper_params))
        #clone estimators
        #Clfs = [clone(est).set_params(**p) for p in param_grid]
        Clfs = [copy.deepcopy(est).set_params(**p) for p in param_grid]
#        results=[]
        
        


        for clf in Clfs:

            # fix the seed of the estimator if it has one
            for a in ['random_state','seed']:
                if hasattr(clf,a):
                    setattr(clf,a,random_state)
            if (est_name=='GPMaLClassifier'):
              setattr(clf,'dataset',dataset)
            print('running',clf.get_params(),'...')
                    # clf.random_state = random_state
            # get the CV score on the training data
            start=time()
            clf.fit(X_train, y_train)
            end=time()
            runtime=(end-start)/3600
            # refit the model to all the data

            # get a holdout test score
            train_bal_accuracy = balanced_accuracy_score(y_train, clf.predict(X_train))
            test_bal_accuracy = balanced_accuracy_score(y_test, clf.predict(X_test))

            model=[]
            best_fitness=0

            if(est_name=='ManiGPClassifier'):
              model=[str(clf.model[0]), str(clf.model[1])]
              best_fitness=clf.best_fitness
            else:
              model=['','']

                            
            results=[{
                   'dataset' : dataname,
                   'seed' : random_state,
                   'algorithm' : est_name,
                   'train_bal_accuracy':train_bal_accuracy,
                   'test_bal_accuracy':test_bal_accuracy,
                   'runtime':  runtime,
                   'parameters': clf.get_params(),
                   'model1' : model[0],
                   'model2' : model[1],
                   'best_fitness' : best_fitness}
                   ]
            # print results
            df = pd.DataFrame.from_records(data=results,columns=results[0].keys())
            df.to_csv(save_file, index=False,header=False,mode='a')
#    df['seed'] = random_state
#    df['dataset'] = dataname
#    df['algorithm'] = est_name
#    df['parameters_hash'] = df['parameters'].apply(lambda x:
#        hash(frozenset(x.items())))
#    print('dataframe columns:',df.columns)
#    print(df[:10])
#    if os.path.isfile(save_file):
#        # if exists, append
#        df.to_csv(save_file, mode='a', header=False, index=False)
#    else:
#        df.to_csv(save_file, index=False)


################################################################################
# main entry point
################################################################################
import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate a method on a dataset.", 
                                     add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG',default=None,type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-save_file', action='store', dest='SAVE_FILE',default=None,
            type=str, help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',default=None,
            type=int, help='Seed / trial')

    args = parser.parse_args()
    # import algorithm 
#    print('import from','methods.'+args.ALG)
    algorithm = importlib.__import__('methods.'+ args.ALG, globals(),locals(),
                                   ['est','hyper_params'])

    #est=ManiGPClassifier()

#    print('algorithm:',algorithm.est)
#    print('hyperparams:',algorithm.hyper_params)

#    text='\t'.join([str(args.INPUT_FILE), ' ', str(args.SAVE_FILE), ' ', str(args.RANDOM_STATE), ' ',
#                   str(algorithm.est), ' ', str(algorithm.hyper_params)])
#    print(text)


    evaluate_model(args.INPUT_FILE, args.SAVE_FILE, args.RANDOM_STATE, 
                   algorithm.est, algorithm.hyper_params)
