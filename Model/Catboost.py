import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载数据
data = pd.read_excel(r"data.xlsx")
print("-----------------------------------------------数据前5行-----------------------------------------------")
print(data.head())
X = data.drop("compressive strength", axis=1)
y = data[["compressive strength"]]  # 标签

print('-----------------------------------------------数据分析-----------------------------------------------')
print(X.describe().T)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def mape(y_true, y_pred):
    n = len(y_true)
    mape_sum = np.sum(np.abs((y_pred - y_true) / y_true))
    return (mape_sum / n) * 100


def wmape(y_true, y_pred):
    wmape_sum = np.sum(np.abs(y_pred - y_true) / y_true * y_true)
    y_sum = np.sum(y_true)
    return wmape_sum / y_sum


def rsr(y_true, y_pred):
    n = len(y_true)
    mean_y = np.mean(y_true)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return np.sqrt((n * rmse) / np.sum((y_true - mean_y) ** 2))


def vaf(y_true, y_pred):
    residual_var = np.var(np.array(y_true) - np.array(y_pred))
    total_var = np.var(y_true)
    return (1 - residual_var / total_var) * 100

# 网格搜索 对CatBoost模型进行超参优化
param_grid_catboost = {
    'iterations': [100, 200, 300],
    'depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2]
}

catboost_model = CatBoostRegressor()
grid_search_catboost = GridSearchCV(estimator=catboost_model, param_grid=param_grid_catboost, cv=5,
                                    scoring='neg_mean_squared_error', verbose=0)
grid_search_catboost.fit(X_train, y_train)

# 获取最佳模型和其参数
best_catboost_model = grid_search_catboost.best_estimator_
best_params_catboost = grid_search_catboost.best_params_
print("Best CatBoost parameters:", best_params_catboost)

# 在测试集上进行预测
y_pred = best_catboost_model.predict(X_test)
y_p_train = best_catboost_model.predict(X_train)
print('-----------------------------------------------预测结果-----------------------------------------------')
print("预测值", y_pred.tolist())
print("真实值", [i[0] for i in y_test.values.tolist()])
print()
train_rmse = mean_squared_error([i[0] for i in y_train.values.tolist()], y_p_train.tolist(),
                                multioutput='uniform_average') ** 0.5
test_rmse = mean_squared_error([i[0] for i in y_test.values.tolist()], y_pred.tolist(),
                               multioutput='uniform_average') ** 0.5
print('Train RMSE: %.5f' % train_rmse)
print('Test RMSE: %.5f' % test_rmse)
print()
train_r2 = r2_score([i[0] for i in y_train.values.tolist()], y_p_train.tolist())
test_r2 = r2_score([i[0] for i in y_test.values.tolist()], y_pred.tolist())
print('Train R2: %.5f' % train_r2)
print('Test R2: %.5f' % test_r2)
print()
train_mae = mean_absolute_error([i[0] for i in y_train.values.tolist()], y_p_train.tolist())
test_mae = mean_absolute_error([i[0] for i in y_test.values.tolist()], y_pred.tolist())
print('Train MAE: %.5f' % train_mae)
print('Test MAE: %.5f' % test_mae)
print()
train_mape = mape(numpy.array([i[0] for i in y_train.values.tolist()]), numpy.array(y_p_train.tolist()))
test_mape = mape(numpy.array([i[0] for i in y_test.values.tolist()]), numpy.array(y_pred.tolist()))
print('Train MAPE: %.5f' % train_mape)
print('Test MAPE: %.5f' % test_mape)
print()
train_rsr = rsr(np.array([i[0] for i in y_train.values.tolist()]), np.array(y_p_train.tolist()))
test_rsr = rsr(np.array([i[0] for i in y_test.values.tolist()]), np.array(y_pred.tolist()))
print('Train RSR: %.5f' % train_rsr)
print('Test RSR: %.5f' % test_rsr)
print()
train_wmape = wmape(np.array([i[0] for i in y_train.values.tolist()]), np.array(y_p_train.tolist()))
test_wmape = wmape(np.array([i[0] for i in y_test.values.tolist()]), np.array(y_pred.tolist()))
print('Train WMAPE: %.5f' % train_wmape)
print('Test WMAPE: %.5f' % test_wmape)
print()
train_vaf = vaf(np.array([i[0] for i in y_train.values.tolist()]), np.array(y_p_train.tolist()))
test_vaf = vaf(np.array([i[0] for i in y_test.values.tolist()]), np.array(y_pred.tolist()))
print('Train VAF: %.5f' % train_vaf + "%")
print('Test VAF: %.5f' % test_vaf + "%")
