import pandas as pd
import statsmodels.api as sm

# 读取广西14市面板数据
data = pd.read_csv('dataset/2015-2022_广西面板数据.csv')

# 构建双重差分模型
did_model = sm.OLS.from_formula(
    'GDP_growth ~ Digital_Index*Policy_Dummy + Urban_Rate + EDU_Level + FDI + C(Year) + C(City)',
    data=data
)

# 输出稳健标准误结果
results = did_model.fit(cov_type='cluster', cov_kwds={'groups': data['City']})
print(results.summary())