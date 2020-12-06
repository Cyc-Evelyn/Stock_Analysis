##01
自訂函數計算收益均值/風險平方差與波動性
"""

def get_mean(num_list): #平均值
        def __get_sum(num_list):
            sum_ = 0
            for i in num_list:
                sum_ += i
            return sum_
        def __get_length(num_list):
            length_ = 0
            for i in num_list:
                length_ += 1
            return length_
        return __get_sum(num_list) / __get_length(num_list)

def get_sq(num_list): #平方差
    return([((i-get_mean(num_list))**2) for i in num_list])
            
 #實測:基金的平方差       
revenue_list = [0, 8, 18, 13, 6, 10]
print(get_sq(revenue_list))

class Invest:
    """
    分類與評估投資標的的收益
    """
    def __init__(self,category,mean,weight,personalrisk): 
        self._category=category #投資類別
        self._mean=mean         #均值
        self._weight=float(weight) #權重
        self._personalrisk=personalrisk #個人風險承受度
   
    def get_revenue(self):
                __total_revenue=self._mean*self._weight
                return "收益為{}元".format(__total_revenue) 
#實測:基金的收益        
revenue_list = [0, 8, 18, 13, 6, 10]
基金均值=get_mean(revenue_list)

基金=Invest("fund",基金均值,0.5,"低")
print(基金.get_revenue())

class Risk(Invest):
    """
    評估投資標的的風險,繼承invest的資料做投資風險分析
    """
    def __init__(self,category,mean,weight,personalrisk,beta): 
        super().__init__(category,mean,weight,personalrisk)
        self._beta=beta  #加入風險波動值
    
    def get_evaluate(self): #投資風險分析
      if self._beta>=1.5:
          evaluate="高"
      else:                                                                                                                                                                                           
          evaluate="低"
      return '投資波動幅度為市場的{}倍,個人承受風險程度為{},為{}風險投資'.format(self._beta,self._personalrisk,evaluate)

#實測:股票的投資風險
revenue_list = [51, -8, 60, -30, 5, 12]
股票均值=get_mean(revenue_list)
股票=(Risk("fund",股票均值,0.5,"高",1.5))
print(股票.get_evaluate())

"""
## 02

使用Pandas爬取yahoofinance的股價並加以分析

##簡單報酬率*2
"""

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
tickers=['PG','MSFT','F','GE']
mydata=pd.DataFrame()
for t in tickers:
    mydata[t]=wb.DataReader(t,data_source='yahoo',start='2010-1-1')['Adj Close']
(mydata/mydata.iloc[0]*100).plot(figsize=(15,6));
plt.show()
returns=(mydata/mydata.shift(1))-1
weights=np.array([0.25,0.25,0.25,0.25])
annual_returns=returns.mean()*250
pfo=str(round(np.dot(annual_returns,weights),5)*100)+'%'
print(pfo)

ind_tickers=['^GSPC', '^IXIC', '^GDAXI', '^VIX']
ind_data=pd.DataFrame()
for t in ind_tickers:
    ind_data[t]=wb.DataReader(t,data_source='yahoo',start='2010-1-1')['Adj Close']
(ind_data/ind_data.iloc[0]*100).plot(figsize=(15,6));
plt.show()
ind_returns=(ind_data/ind_data.shift(1))-1
weights=np.array([0.25,0.25,0.25,0.25])
annual_ind_returns=ind_returns.mean()*250
ind_pfo=str(round(np.dot(annual_ind_returns,weights),5)*100)+'%'
print(ind_pfo)

"""
##指數報酬率
##投資組合分散風險
"""

logind_returns=np.log(ind_data/ind_data.shift(1))
weights=np.array([0.25,0.25,0.25,0.25])
annual_logind_returns=logind_returns.mean()*250
logind_pfo=str(round(np.dot(annual_logind_returns,weights),5)*100)+'%'
print(logind_pfo)

#比較均報酬
print(logind_returns['^VIX'].mean())
print(logind_returns['^GSPC'].mean())

#比較標準差
print(logind_returns['^VIX'].std()*250**0.5)
print(logind_returns['^GSPC'].std()*250**0.5)

#比較變異數
VIX_VAR_a=logind_returns['^VIX'].var()*250
GDAXI_VAR_a=logind_returns['^GDAXI'].var()*250
IXIC_VAR_a=logind_returns['^IXIC'].var()*250
GSPC_VAR_a=logind_returns['^GSPC'].var()*250
print(VIX_VAR_a)

#(是否可藉投資組合分擔風險,相關程度的衡量)
#相關係數
corr_matrix=logind_returns.corr()
print(corr_matrix)

"""##系統性與非系統性風險"""

#可分散的非系統風險dr
pfo_var=np.dot(weights.T,np.dot(logind_returns.cov()*250,weights))
pfo_vol=(pfo_var)**0.5
print(pfo_var)
print(pfo_vol)
dr=pfo_var-(weights[0]**2*GSPC_VAR_a)-(weights[1]**2*IXIC_VAR_a)-(weights[2]**2*GDAXI_VAR_a)-(weights[3]**2*VIX_VAR_a)
print(str(round(dr*100,3))+'%') #可分散的風險值 轉成百分比形式
#不可分散的系統風險ndr
ndr=pfo_var-dr
print(ndr) #不可分散風險的部分

"""
##蒙特卡羅模擬估算不同投資比率下的風險
"""

num_assets=len(ind_tickers)#投資組合內有幾個投資標的

pfo_return=[]#組合收益
pfo_volatility =[]#組合波動性(越小越好)
for i in range(1000):
    weight_ran=np.random.random(num_assets)#隨機生成投資比例
    weight_ran/=np.sum(weight_ran)#確保投資比例總合為1
    pfo_return.append(np.sum(weight_ran*logind_returns.mean())*250)
    pfo_volatility.append(np.sqrt(np.dot(weight_ran.T,np.dot(logind_returns.cov()*250,weight_ran))))
pfo_return=np.array(pfo_return)
pfo_volatility=np.array(pfo_volatility)
pfo_table=pd.DataFrame({"收益":pfo_return,"風險":pfo_volatility})#生成1000筆投資組合表格
pfo_table.plot(x="風險",y="收益",kind='scatter',figsize=(15,6));
print(pfo_table)
plt.xlabel("predict risk")
plt.ylabel("predict pfo")
plt.show()#使用散佈圖觀察最佳風險與收益的組合
