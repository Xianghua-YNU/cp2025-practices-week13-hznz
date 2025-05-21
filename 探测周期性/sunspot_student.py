#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 完整实现
"""

import numpy as np
import matplotlib.pyplot as plt

def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据
    
    参数:
        url (str): 本地文件路径
        
    返回:
        tuple: (years, sunspots) 年份和太阳黑子数
    """
    # 加载数据，选择第1列（年份）和第2列（太阳黑子数）（假设列索引从0开始）
    data = np.loadtxt(url, usecols=(1, 2))
    years = data[:, 0]  # 年份列
    sunspots = data[:, 1]  # 太阳黑子数列
    
    # 处理可能的负值（假设数据中存在负数为异常值，替换为0）
    sunspots[sunspots < 0] = 0
    return years, sunspots

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图
    
    参数:
        years (numpy.ndarray): 年份数组
        sunspots (numpy.ndarray): 太阳黑子数数组
    """
    plt.figure()
    plt.plot(years, sunspots, color='blue')
    plt.title("Sunspot Number Over Time")
    plt.xlabel("Year")
    plt.ylabel("Sunspot Number")
    plt.grid(True)
    plt.show()

def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱
    
    参数:
        sunspots (numpy.ndarray): 太阳黑子数数组
        
    返回:
        tuple: (frequencies, power) 频率数组和功率谱
    """
    N = len(sunspots)
    fft_coeff = np.fft.fft(sunspots)  # 傅里叶变换
    power = np.abs(fft_coeff) ** 2  # 功率谱
    
    # 生成频率数组（单位：周期/月）
    frequencies = np.fft.fftfreq(N, d=1)[:N//2]  # 只取正频率部分
    
    # 对应功率谱部分（排除负频率和零频）
    power = power[:N//2]
    frequencies = frequencies[1:]  # 排除零频（k=0）
    power = power[1:]
    
    return frequencies, power

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
    """
    plt.figure()
    plt.plot(1/frequencies, power, color='red')  # 横轴为周期（月）
    plt.title("Power Spectrum of Sunspot Data")
    plt.xlabel("Period (months)")
    plt.ylabel("Power")
    plt.xlim(0, 200)  # 限制周期范围以突出主峰
    plt.grid(True)
    plt.show()

def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
        
    返回:
        float: 主周期（月）
    """
    # 找到最大功率对应的索引（排除零频后）
    max_idx = np.argmax(power)
    main_frequency = frequencies[max_idx]
    main_period = 1 / main_frequency  # 周期 = 1/频率
    return main_period

def main():
    # 数据文件路径
    data = "sunspot_data.txt"
    
    # 1. 加载并可视化数据
    years, sunspots = load_sunspot_data(data)
    plot_sunspot_data(years, sunspots)
    
    # 2. 傅里叶变换分析
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)
    
    # 3. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"\nMain period of sunspot cycle: {main_period:.2f} months")
    print(f"Approximately {main_period/12:.2f} years")

if __name__ == "__main__":
    main()
