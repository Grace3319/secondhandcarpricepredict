#-*- codeing = utf-8 -*-
#@Time : 2021/6/24 14:40
#@Author : Grace
#@File : 1.py
#@Software: PyCharm

##先运行

import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
import time

warnings.filterwarnings('ignore')
import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt
## 模型预测的
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
Train_data = pd.read_csv('venv/Data/used_car_train_20200313.csv', sep=' ')
TestB_data = pd.read_csv('venv/Data/used_car_testB_20200421.csv', sep=' ')

## 输出数据的大小信息
print('Train data shape:',Train_data.shape)
print('TestB data shape:',TestB_data.shape)
print(Train_data.head())#看头五行
print(Train_data.info())#通过 .info() 简要可以看到对应一些数据列名，以及NAN缺失信息
## 通过 .columns 查看列名
print(Train_data.columns)
print(TestB_data.info())
print(Train_data.describe())#可以查看数值特征列的一些统计信息

###特征与标签构建
numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
categorical_cols = Train_data.select_dtypes(include = 'object').columns

## 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID','name',
                                                             'regDate','creatDate'
    ,'price','model','brand','regionCode','seller']]
feature_cols = [col for col in feature_cols if 'Type' not in col]

## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
Y_data = Train_data['price']

X_test  = TestB_data[feature_cols]

## 定义了一个统计函数，方便后续信息统计
def Sta_inf(data):
    print('_min',np.min(data))
    print('_max:',np.max(data))
    print('_mean',np.mean(data))
    print('_ptp',np.ptp(data))
    print('_std',np.std(data))
    print('_var',np.var(data))

#统计标签的基本分布信息
print('Sta of label:')
Sta_inf(Y_data)

#缺省值用-1填补
X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)

## xgb-Model
xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0,
                       subsample=0.8, \
                       colsample_bytree=0.9, max_depth=7)
scores_train = []
scores = []

## 5折交叉验证方式
sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_ind, val_ind in sk.split(X_data, Y_data):
    train_x = X_data.iloc[train_ind].values
    train_y = Y_data.iloc[train_ind]
    val_x = X_data.iloc[val_ind].values
    val_y = Y_data.iloc[val_ind]

    xgr.fit(train_x, train_y)
    pred_train_xgb = xgr.predict(train_x)
    pred_xgb = xgr.predict(val_x)

    score_train = mean_absolute_error(train_y, pred_train_xgb)
    scores_train.append(score_train)
    score = mean_absolute_error(val_y, pred_xgb)
    scores.append(score)

print('Train mae:', np.mean(score_train))
print('Val mae', np.mean(scores))