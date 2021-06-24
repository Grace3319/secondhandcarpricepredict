#-*- codeing = utf-8 -*-
#@Time : 2021/6/24 14:40
#@Author : Grace
#@File : 2.py
#@Software: PyCharm


import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
import time

warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
## 模型预测的
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
# 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA
import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import D

X_data,Y_data,X_test=D.X_data,D.Y_data,D.X_test
scores=D.scores
score_train=D.score_train


#定义xgb和lgb模型函数
def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=7) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators = 150)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm


#切分数据集（Train,Val）进行模型训练，评价和预测
x_train,x_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.3)
print('Train lgb...')
model_lgb = build_model_lgb(x_train,y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = mean_absolute_error(y_val,val_lgb)
print('MAE of val with lgb:',MAE_lgb)


print('Predict lgb...')
model_lgb_pre = build_model_lgb(X_data,Y_data)
subA_lgb = model_lgb_pre.predict(X_test)
print('Sta of Predict lgb:')
D.Sta_inf(subA_lgb)


print('Train xgb...')
model_xgb = build_model_xgb(x_train,y_train)
val_xgb = model_xgb.predict(x_val)
MAE_xgb = mean_absolute_error(y_val,val_xgb)
print('MAE of val with xgb:',MAE_xgb)

print('Predict xgb...')
model_xgb_pre = build_model_xgb(X_data,Y_data)
subA_xgb = model_xgb_pre.predict(X_test)
print('Sta of Predict xgb:')
D.Sta_inf(subA_xgb)

##进行两模型的结果加权融合
val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/
                                                      (MAE_xgb+MAE_lgb))*val_xgb
val_Weighted[val_Weighted<0]=10
# 由于发现预测的最小值有负数，而真实情况下，price为负是不存在的，由此我们进行对应的后修正
print('MAE of val with Weighted ensemble:',mean_absolute_error(y_val,
                                                               val_Weighted))

sub_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*subA_lgb+(1-MAE_xgb/
                                                       (MAE_xgb+MAE_lgb))*subA_xgb

#查看预测值的统计进行
plt.hist(Y_data)
plt.show()
plt.close()

sub = pd.DataFrame()
sub['SaleID'] = X_test.index
sub['price'] = sub_Weighted
sub.to_csv('./sub_Weighted.csv',index=False)
#print(sub.head())

#价格与注册时间（creatDate类似分析）：
plt.scatter(D.Train_data.SaleID, D.Train_data.price)
plt.ylabel("price")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title("saleid")
plt.show()

##价格与销售个体
plt.scatter(D.Train_data.seller, D.Train_data.price)
plt.ylabel("price")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title("seller")
plt.show()

#价格与车身类型（燃油类型类似）
plt.scatter(D.Train_data.bodyType, D.Train_data.price)
plt.ylabel("price")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title("bodyType")
plt.show()

#价格与注册时间（creatDate类似分析）
plt.scatter(D.Train_data.regDate, D.Train_data.price)
plt.ylabel("price")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title("regDate")
plt.show()


#价格与车身类型（燃油类型类似）
plt.scatter(D.Train_data.gearbox, D.Train_data.price)
plt.ylabel("price")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title("gearbox")
plt.show()

#不同变速箱的价格分布
D.Train_data.price[D.Train_data.gearbox == 0].plot(kind='kde')
D.Train_data.price[D.Train_data.gearbox == 1].plot(kind='kde')
plt.xlabel("price")# plots an axis lable
plt.ylabel("proba")
plt.legend(('auto:1', 'manual:0'),loc='best')
plt.show()