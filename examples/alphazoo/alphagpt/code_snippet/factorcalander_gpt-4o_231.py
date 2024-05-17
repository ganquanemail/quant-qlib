#encoding=gbk
import numpy as np

class ReversalFrequency:
    def __init__(self, overnight_returns, daytime_returns):
        """
        初始化
        :param overnight_returns: 隔夜收益率列表
        :param daytime_returns: 日内收益率列表
        """
        self.overnight_returns = np.array(overnight_returns)
        self.daytime_returns = np.array(daytime_returns)

    def calc_reversal_frequency(self):
        """
        计算反向日内逆转的频率
        :return: 反向日内逆转的频率
        """
        T = len(self.overnight_returns)
        indicator = (self.overnight_returns > 0) & (self.daytime_returns < 0)
        return np.sum(indicator) / T

# 示例用法
# 隔夜收益率
overnight_returns = [0.01, -0.02, 0.015, -0.007, 0.03, -0.012, 0.025]

# 日内收益率
daytime_returns = [-0.008, 0.01, -0.005, 0.004, -0.02, 0.011, -0.015]

# 计算反向日内逆转的频率
reversal_freq_calculator = ReversalFrequency(overnight_returns, daytime_returns)
frequency = reversal_freq_calculator.calc_reversal_frequency()

print(f"反向日内逆转的频率: {frequency}")
