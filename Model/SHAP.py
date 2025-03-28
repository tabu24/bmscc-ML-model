
import shap

plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

model = best_xgb_model

X = data.drop("compressive strength", axis=1)

explainer = shap.TreeExplainer(model)  # 创建解释器
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

plt.show()
