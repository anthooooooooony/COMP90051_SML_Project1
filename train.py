import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


INSTANCE_LIMIT = 2000

pos_df = pd.read_csv('pos1.csv', sep='\t')
pos_df['label'] = 1
neg_df = pd.read_csv('neg1.csv', sep='\t')
neg_df['label'] = 0
test_final = pd.read_csv('test1.csv', sep='\t')

feature_df = pd.concat([pos_df, neg_df], ignore_index=True)
label = feature_df['label']
label = np.array(label)
print label

columns = [
    'adamic_adar_index',
    'cluster_source',
    'cluster_target',
    'common_predecessor_number',
    'common_successor_number',
    'jaccard_coefficient',
    'jaccard_distance_between_predecessors',
    'jaccard_distance_between_successors',
    'len_source_predecessors',
    'len_source_successors',
    'len_target_predecessors',
    'len_target_successors',
    'preference_attachment',
    'resource_allocation_index',
    #'shortest_path',
    'source_authorities',
    'source_hubs',
    'source_pagerank',
    'target_authorities',
    'target_hubs',
    'target_pagerank'
]

feature_df = feature_df[columns]
test_final = test_final[columns]

X_train, X_test, y_train, y_test = train_test_split(
    feature_df, label, test_size=0.95, random_state=42)


train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 6,
    'max_bin': 6,
    'min_sum_hessian_in_leaf': 3,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.85,
    'bagging_freq': 2,
    'lambda_l1': 0.1,
    'lambda_l2': 0.01,
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'top_k': 10
}
'''
gbm = lgb.LGBMRegressor(boosting_type='dart',
                        num_leaves=31,
                        max_depth=-1,
                        learning_rate=0.01,
                        n_estimators=200,
                        objective='regression',
                        is_unbalance=True,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        min_child_samples=15,
                        min_split_gain=2.
                        )

gbm.fit(X_train, y_train, eval_metric=['logloss', 'auc'], verbose=True)
'''
gbm = lgb.train(params, train_data, categorical_feature=columns)
preds = gbm.predict(X_test, gbm.best_iteration)
print preds[:50]
print metrics.roc_auc_score(y_test, preds)


print gbm.feature_importance()




#input_id = [c for c in range(1, 2001)]
#output = pd.DataFrame({'Id': input_id, 'Prediction': preds})
#output.to_csv('submission.csv', index=False)
