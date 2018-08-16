import pandas as pd  # 数据处理包
import time
import datetime
import operator
import numpy as np
from sklearn.model_selection import ShuffleSplit
from datetime import datetime
from pandas import DataFrame
from sklearn.cross_validation import train_test_split  # 数据分割
from sklearn.feature_extraction import DictVectorizer  # 特征转化器
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升树分类器
from sklearn.linear_model.logistic import LogisticRegression
from xgboost import XGBClassifier  # XGBoost分类器
from sklearn.metrics import log_loss  # logloss损失函数
from sklearn.cross_validation import cross_val_score  # 交叉验证
from sklearn.preprocessing import StandardScaler  # 标准化数据
from sklearn.preprocessing import MinMaxScaler  # 归一化数据
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xg


def fea_choose(log, data):
    ttt = []
    x = log.groupby('USRID').size().reset_index().rename(columns={0: 'user_click'})
    data = pd.merge(data, x, on='USRID', how='left')  # 3月总点击量
    data['biaoji'] = (data['user_click'] > 0).apply(lambda x: 1 if x else 0)

    x = log.groupby(['USRID'])['day'].max().reset_index().rename(columns={'day': 'user_click_last'})
    y = log.groupby(['USRID'])['day'].min().reset_index().rename(columns={'day': 'user_click_first'})
    data = pd.merge(data, x, on='USRID', how='left')  # 最后一天点击
    data = pd.merge(data, y, on='USRID', how='left')  # 第一天点击

    x = log.groupby(['USRID', 'day']).size().reset_index().rename(columns={0: 'user_day_click'})
    y = x.groupby('USRID').size().reset_index().rename(columns={0: 'user_click_day_num'})
    data = pd.merge(data, y, on='USRID', how='left')  # 点击天数


    x = log.groupby(['USRID', 'day']).size().reset_index().rename(columns={0: 'user_day_click'})  # 每天的点击量
    y = x.groupby(['USRID'], as_index=False)['user_day_click'].agg({
        'user_click_mean': np.mean,
        'user_click_std': np.std,
        'user_click_min': np.min,
        'user_click_max': np.max
    })
    data = pd.merge(data, y, on='USRID', how='left')

    x = log.groupby(['USRID', 'hour']).size().reset_index().rename(columns={0: 'user_hour_click'})  # 每小时的点击量
    y = x.groupby(['USRID'], as_index=False)['user_hour_click'].agg({
        'user_hour_click_mean': np.mean,
        'user_hour_click_std': np.std,
        'user_hour_click_min': np.min,
        'user_hour_click_max': np.max
    })
    data = pd.merge(data, y, on='USRID', how='left')

    x = log.groupby(['USRID', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_day_hour_click'})  # 每天每小时的点击量
    y = x.groupby(['USRID'], as_index=False)['user_day_hour_click'].agg({
        'user_day_hour_click_mean': np.mean,
        'user_day_hour_click_std': np.std,
        'user_day_hour_click_min': np.min,
        'user_day_hour_click_max': np.max
    })
    data = pd.merge(data, y, on='USRID', how='left')

	'''
    x = log.groupby(['USRID', 'EVT_LBL']).size().reset_index().rename(columns={0: 'user_EVT_click'})
    y = x.groupby(['USRID'])['EVT_LBL'].size().reset_index().rename(columns={0: 'EVT_LBL_Ca_num'})
    z = x.groupby(['USRID'], as_index=False)['user_EVT_click'].agg({
        'user_EVT_click_mean': np.mean,
        'user_EVT_click_std': np.std,
        'user_EVT_click_min': np.min,
        'user_EVT_click_max': np.max
    })
    data = pd.merge(data, y, on='USRID', how='left')  # EVT_LBL访问种类个数
    data = pd.merge(data, z, on='USRID', how='left')

	
    x = log.groupby(['USRID'])['action_0'].nunique().reset_index()
    y = log.groupby(['USRID'])['action_1'].nunique().reset_index()
    z = log.groupby(['USRID'])['action_2'].nunique().reset_index()
    data = pd.merge(data, x, on='USRID', how='left')
    data = pd.merge(data, y, on='USRID', how='left')
    data = pd.merge(data, z, on='USRID', how='left')
    print(x)
	'''

    df = log[['USRID', 'action_0']]
    x = df['action_0'].drop_duplicates().values
    df = pd.get_dummies(df, columns=['action_0'], prefix='label')
    for i in x:
        y = df.groupby(['USRID'])['label_%d' % i].sum().reset_index()
        ttt.append('label_%d' % i)
        data = pd.merge(data, y, on='USRID', how='left')
    '''
    df = log[['USRID', 'action_1']]
    x = df['action_1'].drop_duplicates().values
    df = pd.get_dummies(df, columns=['action_1'], prefix='label_1')
    for i in x:
        y = df.groupby(['USRID'])['label_1_%d' % i].sum().reset_index()
        # print(y.head(n = 5))
        ttt.append('label_1_%d' % i)
        data = pd.merge(data, y, on='USRID', how='left')
    '''

    df = log[['USRID','action_2']]
    x = df['action_2'].drop_duplicates().values
    df = pd.get_dummies(df,columns=['action_2'],prefix='label_2')
    for i in x:
        y = df.groupby(['USRID'])['label_2_%d'%i].sum().reset_index()
        ttt.append('label_2_%d' % i)
        data = pd.merge(data,y,on = 'USRID',how = 'left')

    df = log[['USRID', 'TCH_TYP']]
    x = df['TCH_TYP'].drop_duplicates().values
    df = pd.get_dummies(df, columns=['TCH_TYP'], prefix='label_T')
    for i in x:
        y = df.groupby(['USRID'])['label_T_%d' % i].sum().reset_index()
        ttt.append('label_T_%d' % i)
        data = pd.merge(data, y, on='USRID', how='left')

    return data, ttt


def time_diff(log, data):
    df = log[['USRID', 'OCC_TIM_t', 'day']]
    df = df.sort_values(by=['USRID', 'OCC_TIM_t']).reset_index()
    df['time_cha'] = df.groupby('USRID')['day'].diff(periods=1)
    df['time_cha'].fillna(0)
    x = df.groupby(['USRID'], as_index=False)['time_cha'].agg({
        'time_cha_mean': np.mean,
        'time_cha_std': np.std,
        'time_cha_min': np.min,
        'time_cha_max': np.max
    })

    data = pd.merge(data, x, on='USRID', how='left')
    df['next_time'] = df.groupby(['USRID'])['OCC_TIM_t'].diff(-1).apply(np.abs)
    y = df.groupby(['USRID'], as_index=False)['next_time'].agg({
        'next_time_mean': np.mean,
        'next_time_std': np.std,
        'next_time_min': np.min,
        'next_time_max': np.max
    })
    data = pd.merge(data, y, on='USRID', how='left')

    return data


def log_tr(log):
    log['action'] = log.EVT_LBL.apply(lambda x: x.split('-'))
    log['time'] = log.OCC_TIM.apply(lambda x: x.split(' '))
    log['time1'] = log['time'].apply(lambda x: x[0].split('-'))
    log['time2'] = log['time'].apply(lambda x: x[1].split(':'))
    log['hour'] = log['time2'].apply(lambda x: x[0]).astype('int')
    log['day'] = log['time1'].apply(lambda x: x[2]).astype('int')
    log['action_0'] = log['action'].apply(lambda x: x[0]).astype('int')
    log['action_1'] = log['action'].apply(lambda x: x[1]).astype('int')
    log['action_2'] = log['action'].apply(lambda x: x[2]).astype('int')
    log['OCC_TIM_t'] = log['OCC_TIM'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

    return log


if __name__ == '__main__':
    # 读取用户信息
    online = 2
    train_agg = pd.read_csv('H:/ZhaoY/train/train_agg.csv', sep='\t')
    x1 = train_agg['USRID'].values
    test_agg = pd.read_csv('H:/ZhaoY/test/test_agg.csv', sep='\t')
    x2 = test_agg['USRID'].values
    agg = pd.concat([train_agg, test_agg])

    # 读取用户行为信息
    train_log = pd.read_csv('H:/ZhaoY/train/train_log.csv', sep='\t')
    test_log = pd.read_csv('H:/ZhaoY/test/test_log.csv', sep='\t')
    log = pd.concat([train_log, test_log], axis=0, ignore_index=True)
    log = log_tr(log)

    print(log.shape)

    x = log['USRID'].drop_duplicates().values
    y = [i for i in x if i in x1]
    z = [i for i in x if i in x2]
    print(len(y), len(z))

    # 读取用户标识信息
    train_flg = pd.read_csv('H:/ZhaoY/train/train_flg.csv', sep='\t')
    test_flg = pd.read_csv('H:/ZhaoY/submit_sample.csv', sep='\t')
    del test_flg['RST']
    test_flg['FLAG'] = -1
    flg = pd.concat([train_flg, test_flg])

    fea = pd.read_csv('H:/ZhaoY/team_cut.csv', sep='\t')
    fea1 = pd.read_csv('H:/ZhaoY/team_cut_1.csv', sep='\t')
    print(fea,fea1)

    data = pd.merge(agg, flg, on=['USRID'], how='left')
    data, f = fea_choose(log, data)
    data = time_diff(log, data)
    data = pd.merge(data, fea, on='USRID', how='left')
    data = pd.merge(data, fea1, on='USRID', how='left')
    print(f, len(f))
    print(data)

    train_data = data[data.FLAG != -1]
    print(train_data.FLAG.value_counts())
    y = train_data['FLAG'].values
    del train_data['FLAG']
    test_data = data[data.FLAG == -1]

    feature = train_data.columns
    X_pd = train_data[feature]
    feature = [i for i in feature if i != 'USRID']
    print(feature)
    print(len(feature))
    X = train_data[feature].values
    test = test_data[feature].values
    #X_pd = train_data[feature]
    all_val_pre = pd.DataFrame()
    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=False, random_state=2018)

    if online == 2:
        result = pd.DataFrame()
        result['USRID'] = test_data.USRID
        result = result.reset_index()
        result.pop('index')
        print(result)
        res = pd.DataFrame()
        tmp_auc = []
        x = 0
        for k, (train_in, test_in) in enumerate(skf.split(X, y)):
            begin = time.time()
            print('num:%d' % k)

            x_train, y_train, x_test, y_test = X[train_in], y[train_in], X[test_in], y[test_in]
            X_Pd_val = X_pd.iloc[test_in, :]
            print(train_in,test_in)
            #xgbc = XGBClassifier(max_depth=5, colsample_bytree=0.7, learning_rate=0.02, n_estimators=1000,
            #                     subsample=0.7, \
            #                     seed=2018,tree_method = 'hist')
            '''
            xgbc = xg.XGBClassifier(n_estimators=10000, max_depth=4, learning_rate=0.01, subsample=0.7,
                               colsample_bytree=0.8, scale_pos_weight=25.0,tree_method = 'hist',verbose_eval=500,seed = 2018)
            #xgbc.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc',early_stopping_rounds=100,verbose=1)
            xgbc.fit(x_train, y_train,eval_set=[(x_test, y_test)], eval_metric='auc',early_stopping_rounds=100,verbose=0)
            xgbc_predict = xgbc.predict_proba(x_test, ntree_limit=xgbc.best_iteration)[:, 1]

            #xgbc1 = xg.XGBClassifier(n_estimators=xgbc.best_iteration, max_depth=4, learning_rate=0.01, subsample=0.7,
             #                  colsample_bytree=0.8, scale_pos_weight=25.0,tree_method = 'hist',seed = 2018)
            #xgbc1.fit(X,y,verbose=1)
            #predict = xgbc1.predict_proba(test)[:, 1]
            '''
            lgbc = lgb.LGBMClassifier(num_leaves = 31,colsample_bytree = 0.9,learning_rate = 0.01,n_estimators = 10000,subsample = 0.8, \
                                 objective='binary',silent=0,seed = 2018,scale_pos_weight=25)
            lgbc.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=100,
                     verbose=0)
            #lgbc.fit(x_train, y_train)
            gbc_predict = lgbc.predict_proba(x_test,num_iteration=lgbc.best_iteration_)[:, 1]

            X_Pd_val['pre_y'] = gbc_predict
            print(X_Pd_val['pre_y'])
            all_val_pre = pd.concat([all_val_pre, X_Pd_val])
            print(all_val_pre.head())

            print(roc_auc_score(y_test, gbc_predict))
            tmp_auc.append(roc_auc_score(y_test, gbc_predict))
            x += tmp_auc[k]
            predict = lgbc.predict_proba(test)[:, 1]

            res[k] = predict
            print(res[k])
            print((time.time() - begin) / 60)

        print('val_pre:\n')
        print(all_val_pre['pre_y'])
        all_val_pre.to_csv(r'H:\ZhaoY\feature\X_Pd_val_xgb.csv', index=False, sep='\t')

        print(res)
        res[k + 1] = res.mean(1)
        print(res)
        print(tmp_auc)
        print(x / N)
        result['RST'] = res[k + 1]
        print(result)
        result.to_csv('H:/ZhaoY/feature/result_b_xgb.csv', index=False, sep='\t')
        #res.to_csv('H:/ZhaoY/res5.csv', index=False, sep='\t')
