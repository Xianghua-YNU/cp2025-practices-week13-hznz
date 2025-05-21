#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傅立叶滤波和平滑 - 道琼斯工业平均指数分析
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """
    加载道Jones工业平均指数数据
    
    参数:
        filename (str): 数据文件路径
    
    返回:
        numpy.ndarray: 指数数组
    """
    try:
        data = np.loadtxt(filename)
    except Exception as e:
        raise IOError(f"数据加载失败: {str(e)}")
    return data

def plot_data(data, title="Dow Jones Industrial Average"):
    """
    绘制时间序列数据
    
    参数:
        data (numpy.ndarray): 输入数据数组
        title (str): 图表标题
    
    返回:
        matplotlib.figure.Figure: 生成的图表对象
    """
    fig = plt.figure()
    plt.plot(data, color='blue')
    plt.title(title)
    plt.xlabel("Time (Days)")
    plt.ylabel("Index Value")
    plt.grid(True)
    return fig

def fourier_filter(data, keep_fraction=0.1):
    """
    执行傅立叶变换并滤波
    
    参数:
        data (numpy.ndarray): 输入数据数组
        keep_fraction (float): 保留的傅立叶系数比例
    
    返回:
        tuple: (滤波后的数据数组, 原始傅立叶系数数组)
    """
    fft_coeff = np.fft.rfft(data)  # 实数傅立叶变换
    cutoff = int(len(fft_coeff) * keep_fraction)  # 计算保留系数阈值
    filtered_coeff = fft_coeff.copy()
    filtered_coeff[cutoff:] = 0  # 高频系数归零
    filtered_data = np.fft.irfft(filtered_coeff)  # 逆变换
    return filtered_data, fft_coeff

def plot_comparison(original, filtered, title="Fourier Filter Result"):
    """
    绘制原始数据和滤波结果的比较
    
    参数:
        original (numpy.ndarray): 原始数据数组
        filtered (numpy.ndarray): 滤波后的数据数组
        title (str): 图表标题
    
    返回:
        matplotlib.figure.Figure: 生成的图表对象
    """
    fig = plt.figure()
    plt.plot(original, color='blue', label='Original', alpha=0.6)
    plt.plot(filtered, color='red', label='Filtered', linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time (Days)")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid(True)
    return fig

def main():
    # 任务1：数据加载与可视化
    data = load_data('dow.txt')
    plot_data(data, "Dow Jones Industrial Average - Original Data")
    
    # 任务2：傅立叶变换与滤波（保留前10%系数）
    filtered_10, _ = fourier_filter(data, 0.1)
    plot_comparison(data, filtered_10, "Fourier Filter (Keep Top 10% Coefficients)")
    
    # 任务3：修改滤波参数（保留前2%系数）
    filtered_2, _ = fourier_filter(data, 0.02)
    plot_comparison(data, filtered_2, "Fourier Filter (Keep Top 2% Coefficients)")
    
    plt.show()  # 显示所有图表

if __name__ == "__main__":
    main()
