#encoding=gbk
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

class StockReturnAnalysis:
    def __init__(self, stock_data):
        """
        初始化类，传入股票数据的DataFrame，其中应包含日期(Date)、隔夜收益率(RET.COa)、日内收益率(RET.OCa)列。
        """
        self.stock_data = stock_data

    def calculate_abnormal_frequency(self):
        """
        计算正向日内逆转的异常频率。
        """
        # 计算隔夜收益率和日内收益率
        self.stock_data['RET.COa'] = self.stock_data.groupby('Date')['RET.COa'].diff().fillna(0)
        self.stock_data['RET.OCa'] = self.stock_data.groupby('Date')['RET.OCa'].diff().fillna(0)
        
        # 标记正向日内逆转的交易日
        self.stock_data['Positive_Intraday_Reversal'] = (self.stock_data['RET.COa'] < 0) & (self.stock_data['RET.OCa'] > 0)
        
        # 计算每个月的正向日内逆转频率
        self.stock_data['Month_Year'] = self.stock_data['Date'].dt.to_period('M')
        monthly_data = self.stock_data.groupby(['Month_Year', 'Positive_Intraday_Reversal']).size().unstack(fill_value=0)
        
        # 计算每个月的正向日内逆转频率
        monthly_data['Frequency'] = monthly_data['True'] / monthly_data['True'].sum(axis=1)
        
        # 计算过去12个月的正向日内逆转频率的平均值
        months = monthly_data.index.to_perioddelta()
        past_12_months = monthly_data.iloc[:-1 * (months.max() - months.min())]  # 确保过去12个月的数据
        average_frequency_12_months = past_12_months['Frequency'].mean()
        
        # 计算异常频率
        monthly_data['Abnormal_Frequency'] = monthly_data['Frequency'] - average_frequency_12_months
        
        return monthly_data

# 示例用法
# 假设stock_df是一个DataFrame，包含日期(Date)、隔夜收益率(RET.COa)、日内收益率(RET.OCa)列
# stock_analysis = StockReturnAnalysis(stock_df)
# abn_frequency = stock_analysis.calculate_abnormal_frequency()

