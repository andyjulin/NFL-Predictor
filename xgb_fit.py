import numpy as np
import pandas as pd

import random
import timeit

import xgboost as xgb
from sklearn import cross_validation as cv
from sklearn import metrics as skm


def get_log_loss(row):
    ans = row['Victory']

    if (ans not in [0, 1]):
        print ('Error: value must be 0 or 1')
        raise ValueError('Not of correct class')
        return -1000

    return -ans * np.log(row['Prediction']) - (1 - ans) * np.log(1 - row['Prediction'])

def get_random_seed(random_seed = 0):
    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)
    else:
        random.seed()
        np.random.seed()
        random_seed = sys.maxsize * random.random()
        
def create_enum(df, col_name):
    labels, uniques = pd.factorize(df[col_name])
    enum = dict(zip(df[col_name], labels))
    df[col_name] = labels

    return enum

def get_fit_predictions(dataset):    
    train = dataset.loc[dataset['Victory'] != -1 ]
    pred  = dataset.loc[dataset['Victory'] == -1 ]

    train, watch = cv.train_test_split(train, test_size = 0.2, random_state = 42)

#     print 'Data Points Used - Training:', train.shape[0], ', Watching:', watch.shape[0], ', Predictions:', pred.shape[0]
        
    opp_cols = ['Opponent ' + c for c in cols]
    features = np.concatenate([['Home', 'Team', 'Opponent'], cols, opp_cols])
        
    start = timeit.default_timer()

    final_xgb_df = fit_xgb_model(train, watch, pred, 
                                 params, features,
                                 max_num_rounds = 10000,
                                  #use_early_stopping = False, 
                                )

    end = timeit.default_timer()

#     print 'Fit Time:', round(end - start, 2), 'seconds\n'
    
    return pred

# Create the fit function, and related functions


def fit_xgb_model(train, watch, pred, params, features, max_num_rounds = 10,
                  use_early_stopping = True, random_seed = 1):

    get_random_seed(random_seed)
    
    dtrain = xgb.DMatrix(train[features].values, train['Victory'].values)
    dwatch = xgb.DMatrix(watch[features].values, watch['Victory'].values)
    dpred  = xgb.DMatrix(pred[features].values)
            
    xgb_classifier = xgb.train(params, dtrain, num_boost_round = max_num_rounds,
                               evals = [(dtrain, 'train'), (dwatch, 'watch')], 
                               early_stopping_rounds = 100, 
                               verbose_eval = False
                              )
        
    watch['Prediction'] = xgb_classifier.predict(dwatch, ntree_limit = xgb_classifier.best_iteration)
    pred['Prediction']  = xgb_classifier.predict(dpred,  ntree_limit = xgb_classifier.best_iteration)      
    
    log_loss  = watch.apply(lambda row: get_log_loss(row), axis = 1).mean()
    auc_score = skm.roc_auc_score(watch.Victory.values, watch.Prediction.values)
        
#     print 'Log Loss ', round(log_loss,  5), '- AUC Score', round(auc_score, 5)  



params = {
    'learning_rate': 0.02,
    '#subsample': 0.9,
    #'alpha': 10,
    #'lambda': 0.99,
    'gamma': 2.0,
    #'colsample_bytree': 0.7,
    'objective': 'reg:logistic',
    'eval_metric': 'logloss',
    #'max_delta_step': 1.5,
    #'max_depth': 15,
    #'min_child_weight': 2,
}

cols = np.array([
# 'Passing Rank',
# 'Completions',
# 'Attempts',
# 'Completion %',  # Very Bad!
# 'Passing Yards',
# 'Passing Yards / Attempt',
# 'Yards / Reception',
# 'Passing Touchdowns', # Not Helpful
# 'Interceptions', # Not Helpful
'QB Rating',
# 'Sacks', # Not Helpful
# 'Passing Yardage Lost to Sacks',
# 'First Downs Passing', # Okay at best
# 'Fumbles', # Okay at best
# 'Fumbles Lost', # Bad!
'Rushing Rank',
# 'Attempts.1',
'Rushing Yards',
# 'Rushing Yards / Attempt', # Okay at best
# 'Rushing Touchdowns', # Not helpful
'First Downs Rushing',
# 'Kicking Rank',
# 'Kicking Points',
# 'Field Goals Made',
# 'Field Goals Attempted',
# 'Field Goal %',
# 'Longest Field Goal',
# 'Field Goals Blocked',
# 'Extra Points Made',
# 'Extra Points Attempted',
# 'Extra Point %',
# 'Extra Points Blocked',
# 'Punting Rank',
# 'Punts',
# 'Punt Yards',
# 'Punt Yards Avg',
# 'Returning Rank',
# 'Kickoff Returns',
# 'Kickoff Return Yards',
# 'Kickoff Return Avg',
# 'Kickoff Return Touchdowns',
# 'Kickoff Return Long',
# 'Punt Returns',
# 'Punt Return Yards',
# 'Punt Return Avg',
# 'Punt Return Touchdowns',
# 'Punt Return Long',
# 'Defense Rank',
'Points Allowed',
# 'Total Tackles',
# 'Solo Tackles',
# 'Assisted Tackles',
# 'Sacks.1', # Not Helpful?
# 'Sack Yards',
# 'Interceptions.1',
# 'Interception Return Touchdowns',
# 'Forced Fumbles', # Not Helpful
# 'Fumbles Opp Recovered',
# 'Fumbles Opp Recovered Touchdowns',
# 'Passes Defended', # Minor at best
# 'Stuffs',
# 'Stuff Yards',
# 'Safeties',
# 'Downs Rank',
# 'First Downs', # Okay at best
# 'First Downs Rushing.1', # Okay at best (with Passing)
# 'First Downs Passing.1', # Okay at best (with Rushing)
# 'Third Downs Made', # Very bad! (with Fourth Downs)
# 'Third Downs Attempted',
# 'Third Down %',
# 'Fourth Downs Made', # Not Helpful (with Third Downs)
# 'Fourth Downs Attempted',
# 'Fourth Down %',
# 'Penalties', # Very Bad!
# 'First Downs Penalty',
# 'Penalty Yards', # REALLY BAD!
# 'Yardage Rank',
# 'Yards', # REALLY BAD!
# 'Rushing Yards.1',
# 'Rushing Yards / Attempt.1',
# 'Rushing Touchdowns.1',
# 'Passing Yards.1',
# 'Passing Yards / Attempt.1',
# 'Yards / Reception.1',
# 'Passing Touchdowns.1',
# 'Kickoff Return Yards.1',
# 'Punt Return Yards.1',
# 'Penalty Yards.1',
# 'Turnovers Rank',
'Turnover +/-',
# 'Interceptions.2', # Not Helpful
# 'Fumbles Opp Recovered.1',
# 'Interceptions.3',
# 'Fumbles Lost.1',
'Wins',
'Losses',
])

