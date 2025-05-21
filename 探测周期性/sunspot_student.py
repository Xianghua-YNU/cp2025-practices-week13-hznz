#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 修正实现
"""

import numpy as np
import matplotlib.pyplot as plt

def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据
    """
    # 加载数据，假设第1列为年份，第2列为太阳黑子数
    data = np.loadtxt(url, usecols=(1, 2))
    years = data[:, 0]
    sunspots = data[:, 1]
    # 处理异常值（负数替换为0）
    sunspots[sunspots < 0] = 0
    return years, sunspots

def compute_power_spectrum(sunspots):
    """
    计算功率谱（包含去趋势处理）
    """
    N = len(sunspots)
    # 去趋势：减去均值以消除直流分量
    detrended = sunspots - np.mean(sunspots)
    # 傅里叶变换
    fft_coeff = np.fft.fft(detrended)
    # 计算功率谱（取模的平方）
    power = np.abs(fft_coeff) ** 2
    # 生成频率数组（单位：周期/月）
    frequencies = np.fft.fftfreq(N, d=1)[:N//2]  # 取正频率部分
    # 排除零频（索引0）
    return frequencies[1:], power[1:N//2]

def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期（限制在10-12年范围内）
    """
    # 定义合理周期范围（10年=120个月，12年=144个月）
    min_period = 120  # 月
    max_period = 144  # 月
    min_freq = 1 / max_period  # 对应144个月的频率
    max_freq = 1 / min_period  # 对应120个月的频率
    
    # 筛选符合范围的频率
    valid_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    valid_freq = frequencies[valid_mask]
    valid_power = power[valid_mask]
    
    if len(valid_power) == 0:
        return np.nan  # 若无有效频率，返回NaN
    
    # 找到最大功率对应的频率
    max_idx = np.argmax(valid_power)
    main_frequency = valid_freq[max_idx]
    main_period = 1 / main_frequency
    return main_period

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子时间序列
    """
    plt.figure(figsize=(10, 4))
    plt.plot(years, sunspots, color="navy", linewidth=1)
    plt.title("Sunspot Number Over Time (1749-2023)")
    plt.xlabel("Year")
    plt.ylabel("Sunspot Number")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图（横轴为周期）
    """
    periods = 1 / frequencies  # 频率转换为周期（月）
    plt.figure(figsize=(10, 4))
    plt.plot(periods, power, color="crimson", linewidth=1)
    plt.title("Power Spectrum of Sunspot Data")
    plt.xlabel("Period (months)")
    plt.ylabel("Power")
    plt.xlim(100, 150)  # 聚焦主周期范围
    plt.axvline(132, color="gray", linestyle="--", label="11-year cycle")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def main():
    # 加载数据
    years, sunspots = load_sunspot_data("sunspot_data.txt")
    
    # 绘制时间序列
    plot_sunspot_data(years, sunspots)
    
    # 计算功率谱
    frequencies, power = compute_power_spectrum(sunspots)
    
    # 绘制功率谱
    plot_power_spectrum(frequencies, power)
    
    # 检测主周期
    main_period = find_main_period(frequencies, power)
    print(f"Main Period: {main_period:.1f} months ({main_period/12:.2f} years)")

if __name__ == "__main__":
    main()
