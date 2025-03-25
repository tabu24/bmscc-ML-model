import numpy
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载数据
data = pd.read_excel(r"data.xlsx")
print("-----------------------------------------------数据前5行-----------------------------------------------")
print(data.head())
X = data.drop("compressive strength", axis=1)
y = data[["compressive strength"]]  # 标签

print('-----------------------------------------------数据分析-----------------------------------------------')
print(X.describe().T)

# 加载模型 XGB和指标 r2 rmse mae mape
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 网格搜索 对XGB模型进行超惨优化
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2]
}

xgb_model = XGBRegressor()
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                           verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_xgb_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best parameters:", best_params)
# xgb_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = best_xgb_model.predict(X_test)
y_p_train = best_xgb_model.predict(X_train)
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

plt.plot(np.array(y_pred.tolist()), c="red")
plt.plot(np.array(y_test.values), c="blue")
plt.legend(["预测值", "真实值"])
plt.show()

# 计算特征之间的相关系数矩阵
correlation_matrix = X.corr()
# 计算特征重要性
feature_importance = pd.Series(best_xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)

import shap

plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

model = best_xgb_model

X = data.drop("抗压强度", axis=1)

explainer = shap.TreeExplainer(model)  # 创建解释器
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

plt.show()
