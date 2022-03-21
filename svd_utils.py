
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import numpy as np
import pandas as pd

## Processing

def extend_holdout_with_train(train, holdout):
    train_= train[train['movieid'].isin(holdout['movieid'])]    
    t_uniq = train[~train['movieid'].isin(holdout['movieid'])]

    n_uniq = t_uniq['movieid'].nunique()
    
    dict_update = {}
  
    dict_update['movieid'] = list(t_uniq['movieid'].unique())
    dict_update['userid'] = holdout[holdout['movieid'].isin(train['movieid'])]['userid'][:n_uniq]  
    dict_update['rating'] = 0
    
    return holdout[holdout['movieid'].isin(train['movieid'])].append(pd.DataFrame.from_dict(dict_update))

def get_data_description(data_index):
    return dict(
        users = data_index['users'].name,
        items = data_index['items'].name,
        feedback = 'rating',
        n_users = len(data_index['users']),
        n_items = len(data_index['items']),
    )

def matrix_from_observations(data, data_description):
    useridx = data[data_description['users']]
    itemidx = data[data_description['items']]
    values = data[data_description['feedback']]
    return csr_matrix((values, (useridx, itemidx)), dtype='f8')

## Building model

def build_svd_model(config, data, data_description):
    source_matrix = matrix_from_observations(data, data_description)
    _, s, vt = svds(source_matrix, k=config['rank'], return_singular_vectors='vh')
    singular_values = s[::-1]
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors, singular_values

def build_normed_svd_model(config, data, data_description):
    source_matrix = matrix_from_observations(data, data_description)
    _, s, vt = svds(source_matrix.dot(config['norm']), k=config['rank'], return_singular_vectors='vh')
    singular_values = s[::-1]
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors, singular_values

## Evaluation

def simple_model_recom_func(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations

def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]

def svd_model_scoring(params, data, data_description):
    item_factors, sigma = params
    test_matrix = matrix_from_observations(data, data_description)
    
    return test_matrix @ item_factors @ item_factors.T

# File writing

def write_ans_to_file(rec_matrix, data_index, filename='baseline.csv'):
    answers = pd.read_csv('data/sample_submission.csv')
    
    users_to_recs = [' '.join(str(data_index['items'][recs])) for recs in rec_matrix]
    
    for i, _ in enumerate(users_to_recs):
        answers.loc[answers.userid == data_index['users'][i], 'movieid'] = users_to_recs[i]
    
    answers.to_csv(filename, index=False)
