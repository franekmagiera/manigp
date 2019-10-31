import pandas as pd
import numpy as np
import pdb
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler

def read_file(filename, classification=True, label='target', sep='\t'):
    
    if filename.split('.')[-1] == 'gz':
        compression = 'gzip'
    else:
        compression = None

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression,
                engine='python')

    feature_names = np.array([x for x in input_data.columns.values if x != label])

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])

    X = pd.DataFrame(RobustScaler().fit_transform(X))
    # if classes aren't labelled sequentially, fix
    if classification:
        y = LabelEncoder().fit_transform(y)

    return X, y, feature_names

