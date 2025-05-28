#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉格朗日点 - 地球-月球系统L1点位置计算

本模块实现了求解地球-月球系统L1拉格朗日点位置的数值方法。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# 物理常数
G = 6.674e-11   # 万有引力常数 (m³·kg⁻¹·s⁻²)
M = 5.974e24    # 地球质量 (kg)
m = 7.348e22    # 月球质量 (kg)
R = 3.844e8     # 地月平均距离 (m)
omega = 2.662e-6  # 月球绕地球的角速度 (s⁻¹)

def lagrange_equation(r):
    """
    计算L1拉格朗日点平衡方程的值：地球引力 - 月球引力 - 离心力 = 0
    
    参数:
        r (float): 从地心到L1点的猜测距离 (m)
    
    返回:
        float: 方程左右两边的差值，当r为L1点位置时接近零
    """
    term_earth = G * M / (r**2)          # 地球引力项
    term_moon = G * m / ((R - r)**2)     # 月球引力项
    term_centrifugal = omega**2 * r    # 离心力项
    return term_earth - term_moon - term_centrifugal

def lagrange_equation_derivative(r):
    """
    计算L1方程对r的导数，用于牛顿法
    
    参数:
        r (float): 从地心到L1点的猜测距离 (m)
    
    返回:
        float: 方程的导数值
    """
    d_earth = -2 * G * M / (r**3)       # 地球引力项导数
    d_moon = -2 * G * m / ((R - r)**3)   # 月球引力项导数
    d_centrifugal = omega**2         # 离心力项导数
    return d_earth + d_moon - d_centrifugal

def newton_method(f, df, x0, tol=1e-8, max_iter=100):
    """
    牛顿法求解方程f(x)=0
    
    参数:
        f (callable): 目标方程函数
        df (callable): 目标方程的导数函数
        x0 (float): 初始猜测值
        tol (float): 收敛容差，当|f(x)| < tol时停止
        max_iter (int): 最大迭代次数
    
    返回:
        tuple: (近似解, 迭代次数, 是否收敛)
    """
    x = x0
    iterations = 0
    converged = False
    
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:  # 检查是否满足收敛条件
            converged = True
            iterations = i + 1
            break
        
        dfx = df(x)
        if abs(dfx) < 1e-14:       # 避免除零错误
            break
        
        delta = fx / dfx
        x1 = x - delta         # 牛顿迭代公式
        
        if abs(delta / x) < tol:  # 附加收敛条件（解的变化量）
            converged = True
            iterations = i + 1
            x = x1
            break
        
        x = x1
        iterations = i + 1
    
    return x, iterations, converged

def secant_method(f, a, b, tol=1e-8, max_iter=100):
    """
    弦截法求解方程f(x)=0
    
    参数:
        f (callable): 目标方程函数
        a (float): 初始区间左端点
        b (float): 初始区间右端点
        tol (float): 收敛容差，当|f(x)| < tol时停止
        max_iter (int): 最大迭代次数
    
    返回:
        tuple: (近似解, 迭代次数, 是否收敛)
    """
    fa, fb = f(a), f(b)
    
    # 检查初始区间是否有效（函数值需异号）
    if fa * fb > 0:
        print("警告: 区间端点函数值同号，弦截法可能不收敛")
    
    x_prev, x = a, b
    f_prev, f_x = fa, fb
    iterations = 0
    converged = False
    
    for i in range(max_iter):
        if abs(f_x - f_prev) < 1e-14:  # 避免除零错误
            break
        
        # 计算斜率近似值并更新解
        delta_x = x - x_prev
        delta_f = f_x - f_prev
        x_new = x - f_x * delta_x / delta_f
        
        # 更新变量
        x_prev, x = x, x_new
        f_prev, f_x = f_x, f(x)
        iterations += 1
        
        # 检查收敛条件
        if abs(f_x) < tol or abs(x - x_prev) < tol:
            converged = True
            break
    
    return x, iterations, converged

def plot_lagrange_equation(r_min, r_max, num_points=1000):
    """
    绘制L1方程的函数图像以辅助分析
    
    参数:
        r_min (float): 绘图范围最小值 (m)
        r_max (float): 绘图范围最大值 (m)
        num_points (int): 采样点数
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    r_values = np.linspace(r_min, r_max, num_points)
    f_values = np.array([lagrange_equation(r) for r in r_values])
    
    # 寻找零点附近的位置
    zero_crossings = np.where(np.diff(np.signbit(f_values)))[0]
    r_zeros = []
    for idx in zero_crossings:
        r1, r2 = r_values[idx], r_values[idx + 1]
        f1, f2 = f_values[idx], f_values[idx + 1]
        # 线性插值找到更精确的零点
        r_zero = r1 - f1 * (r2 - r1) / (f2 - f1)
        r_zeros.append(r_zero)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_values / 1e8, f_values, label='L1 Equation: $f(r) = \\frac{GM}{r^2} - \\frac{Gm}{(R-r)^2} - \\omega^2 r$')
    # 标记零点
    for r_zero in r_zeros:
        ax.plot(r_zero / 1e8, 0, 'ro', label=f'Zero point: {r_zero:.4e} m')
    # 添加水平和垂直参考线
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    # 设置坐标轴标签和标题
    ax.set_xlabel('Distance from Earth [m]')
    ax.set_ylabel('Equation value')
    ax.set_title('L1 Lagrange Point Equation Balance')
    ax.grid(True, alpha=0.3)# 添加网格
    plt.legend()
    return fig

def main():
    """
    主函数流程：
    1. 绘制方程图像确定初始猜测值
    2. 使用牛顿法和弦截法求解
    3. 与SciPy结果对比验证
    """
    # 步骤1：绘制方程图像
    fig = plot_lagrange_equation(3.0e8, 3.8e8)
    plt.savefig('lagrange_equation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 步骤2：牛顿法求解
    print("\n=== 牛顿法求解 ===")
    initial_guess = 3.5e8  # 初始猜测值（约地月距离的91%）
    solution, iters, conv = newton_method(lagrange_equation, lagrange_equation_derivative, initial_guess)
    if conv:
        print(f"解: {solution:.6e} m")
        print(f"迭代次数: {iters}")
        print(f"相对地月距离比例: {solution/R:.4f}")
        print(f"方程值验证: {lagrange_equation(solution):.2e}")
    else:
        print("未收敛！")
    
    # 步骤3：弦截法求解
    print("\n=== 弦截法求解 ===")
    a, b = 3.2e8, 3.7e8  # 初始区间（地球到月球的80%~94%）
    solution_secant, iters_secant, conv_secant = secant_method(lagrange_equation, a, b)
    if conv_secant:
        print(f"解: {solution_secant:.6e} m")
        print(f"迭代次数: {iters_secant}")
        print(f"相对地月距离比例: {solution_secant/R:.4f}")
        print(f"方程值验证: {lagrange_equation(solution_secant):.2e}")
    else:
        print("未收敛！")
    
    # 步骤4：使用SciPy验证
    print("\n=== SciPy验证 ===")
    solution_scipy = optimize.fsolve(lagrange_equation, 3.5e8)[0]
    print(f"SciPy解: {solution_scipy:.6e} m")
    print(f"相对误差（牛顿法）: {(solution - solution_scipy)/solution_scipy*100:.6f}%")
    print(f"相对误差（弦截法）: {(solution_secant - solution_scipy)/solution_scipy*100:.6f}%")

if __name__ == "__main__":
    main()
