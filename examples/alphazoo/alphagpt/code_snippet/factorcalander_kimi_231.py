#encoding=gbk

class StockMarketAnalysis:
    def __init__(self, stock_data):
        """
        初始化函数，接收股票数据。
        :param stock_data: DataFrame, 包含股票隔夜收益率和日内收益率的数据。
        """
        self.stock_data = stock_data

    def calculate_reverse_intraday_reversal_frequency(self):
        """
        计算反向日内逆转的频率。
        :return: Series, 每只股票的反向日内逆转频率。
        """
        # 隔夜收益率和日内收益率
        overnight_returns = self.stock_data['RET.CO.a']
        intraday_returns = self.stock_data['RET.OC.d']

        # 计算反向日内逆转的频率
        reverse_intraday_reversal = (
            (overnight_returns > 0) & (intraday_returns < 0)
        ).cumsum()

        # 计算每只股票的反向日内逆转频率
        reverse_intraday_reversal_frequency = reverse_intraday_reversal / self.stock_data['T']

        return reverse_intraday_reversal_frequency

# 示例使用
# 假设stock_data是一个包含股票隔夜收益率和日内收益率的DataFrame
# stock_data = pd.read_csv('stock_data.csv')  # 加载数据
# analysis = StockMarketAnalysis(stock_data)
# frequencies = analysis.calculate_reverse_intraday_reversal_frequency()
# print(frequencies)
