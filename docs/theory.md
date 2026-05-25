# 理论文档 / Theoretical Documentation

本文档详细描述倾转旋翼无人机纵向动力学模型、强化学习方法及环境设计。

This document provides a detailed description of the tailsitter UAV longitudinal dynamics model, reinforcement learning methods, and environment design.

---

## 目录 / Table of Contents

1. [坐标系与状态定义](#1-坐标系与状态定义)
2. [动力学模型](#2-动力学模型)
3. [气动力模型](#3-气动力模型)
4. [推进系统模型](#4-推进系统模型)
5. [执行器模型](#5-执行器模型)
6. [强化学习环境](#6-强化学习环境)
7. [奖励函数](#7-奖励函数)
8. [归一化方法](#8-归一化方法)
9. [强化学习算法](#9-强化学习算法)

---

## 1. 坐标系与状态定义

### 1.1 机体坐标系

采用标准机体坐标系（Body Frame）：
- **x 轴**：沿机身纵轴向前（机头方向）
- **z 轴**：垂直机身向下（右手定则）
- **原点**：飞行器质心

### 1.2 状态向量

飞行器纵向动力学状态向量为 7 维：

$$
\mathbf{x} = \begin{bmatrix} u \\ w \\ V \\ \alpha \\ \theta \\ q \\ h \end{bmatrix}
$$

| 符号 | 维度 | 说明 | 单位 |
|------|------|------|------|
| $u$ | 0 | 机体 x 轴速度 | m/s |
| $w$ | 1 | 机体 z 轴速度 | m/s |
| $V$ | 2 | 空速 | m/s |
| $\alpha$ | 3 | 迎角 | rad |
| $\theta$ | 4 | 俯仰角 | rad |
| $q$ | 5 | 俯仰角速度 | rad/s |
| $h$ | 6 | 高度 | m |

### 1.3 运动学关系

空速和迎角由机体速度分量计算：

$$
V = \sqrt{u^2 + w^2}
$$

$$
\alpha = \arctan\left(\frac{w}{u}\right)
$$

当 $V \approx 0$ 时，取 $\alpha = \pi/2$。

---

## 2. 动力学模型

动力学模型逆向工程自 `mdl/Lon_mdl.slx` 中的 Stateflow 脚本。

### 2.1 力的合成

总力在机体坐标系中表示为气动力、推力和重力的矢量和：

$$
\begin{bmatrix} F_x \\ F_z \end{bmatrix} = \mathbf{R}(\alpha) \begin{bmatrix} -L \\ -D \end{bmatrix} + \begin{bmatrix} 2 F_{\text{prop}} \\ 0 \end{bmatrix} + mg \begin{bmatrix} -\sin\theta \\ \cos\theta \end{bmatrix}
$$

其中旋转矩阵：

$$
\mathbf{R}(\alpha) = \begin{bmatrix} -\sin\alpha & \cos\alpha \\ \cos\alpha & \sin\alpha \end{bmatrix}
$$

展开为：

$$
F_x = \sin\alpha \cdot L - \cos\alpha \cdot D + 2F_{\text{prop}} - mg\sin\theta
$$

$$
F_z = -\cos\alpha \cdot L - \sin\alpha \cdot D + mg\cos\theta
$$

### 2.2 状态导数

5-DOF 常微分方程（ODE），包含移动质量耦合：

$$
\dot{q} = \frac{M_y - m_s x_s g \cos\theta - 2 m_s x_s v_s q}{J_y + m_s x_s^2}
$$

$$
\dot{u} = \frac{F_x}{m} - \mu \dot{v}_s
$$

$$
\dot{w} = \frac{F_z}{m} - \mu(-\dot{q} x_s - q v_s)
$$

$$
\dot{\theta} = q
$$

$$
\dot{h} = w \cos\theta - u \sin\theta
$$

其中：
- $m$：总质量，$m = m_b + m_s$
- $m_b$：机体质量（6.3 kg）
- $m_s$：移动质量（0.2 kg）
- $\mu = m_s / m$：质量比（0.0308）
- $J_y$：俯仰惯量（0.183 kg·m²）
- $x_s, v_s, \dot{v}_s$：移动质量的位置、速度、加速度
- $g$：重力加速度（9.81 m/s²）

### 2.3 移动质量简化

在纵向运动模式下，侧滑速度 $u_s = 0$，移动质量状态简化为：

$$
\mathbf{s}_s = \begin{bmatrix} x_s \\ v_s \\ \dot{v}_s \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

简化后的 ODE：

$$
\dot{q} = \frac{M_y}{J_y}
$$

$$
\dot{u} = \frac{F_x}{m}
$$

$$
\dot{w} = \frac{F_z}{m}
$$

### 2.4 数值积分

采用四阶 Runge-Kutta（RK4）方法进行数值积分，时间步长 $T_s = 0.02$ s：

$$
\mathbf{k}_1 = f(\mathbf{x}_n)
$$

$$
\mathbf{k}_2 = f\left(\mathbf{x}_n + \frac{T_s}{2} \mathbf{k}_1\right)
$$

$$
\mathbf{k}_3 = f\left(\mathbf{x}_n + \frac{T_s}{2} \mathbf{k}_2\right)
$$

$$
\mathbf{k}_4 = f\left(\mathbf{x}_n + T_s \mathbf{k}_3\right)
$$

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \frac{T_s}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
$$

---

## 3. 气动力模型

### 3.1 动压

$$
Q = \frac{1}{2} \rho S V^2
$$

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 空气密度 | $\rho$ | 1.225 | kg/m³ |
| 参考面积 | $S$ | 0.62 | m² |
| 弦长 | $c$ | 0.31 | m |

### 3.2 气动系数

基本气动系数由迎角 $\alpha$ 查表获得（来自 `aero_cfg2.mat`）：

$$
C_L(\alpha), \quad C_D(\alpha), \quad C_m(\alpha)
$$

升降舵增量由舵偏角 $\delta_e$ 查表获得（来自 `aerosurface.mat`）：

$$
\Delta C_L(\delta_e), \quad \Delta C_D(\delta_e), \quad \Delta C_m(\delta_e)
$$

总气动系数：

$$
C_L^{\text{total}} = C_L(\alpha) + \Delta C_L(\delta_e)
$$

$$
C_D^{\text{total}} = C_D(\alpha) + \Delta C_D(\delta_e)
$$

$$
C_m^{\text{total}} = C_m(\alpha) + \Delta C_m(\delta_e)
$$

### 3.3 气动力和力矩

$$
L = Q \cdot C_L^{\text{total}} \quad \text{（升力）}
$$

$$
D = Q \cdot C_D^{\text{total}} \quad \text{（阻力）}
$$

$$
M_y^{\text{aero}} = Q \cdot c \cdot C_m^{\text{total}} \quad \text{（俯仰力矩）}
$$

### 3.4 插值方法

使用 `scipy.interpolate.interp1d` 进行线性插值，匹配 Simulink Lookup Table 模块行为。

---

## 4. 推进系统模型

### 4.1 节气门-推力映射

推力由节气门指令 $\delta_t$ 查表获得（来自 `prop.mat`）：

$$
F_{\text{prop}} = f_{\text{interp}}(\delta_t)
$$

其中 $\delta_t \in [0, 1]$，$F_{\text{prop}} \in [0, 7]$ N（单发）。

### 4.2 诱导速度

基于动量理论的诱导速度计算：

$$
v_i = \sqrt{\frac{\max(F_{\text{prop}}, 0)}{2 \rho \pi R^2}}
$$

其中 $R = 0.2032$ m 为螺旋桨半径。

### 4.3 总推力

Simulink 模型中推力系数为 2（双发），总推力为：

$$
F_{\text{total}} = 2 \cdot F_{\text{prop}}
$$

---

## 5. 执行器模型

### 5.1 一阶滞后

节气门和升降舵执行器均采用一阶滞后模型：

$$
u_{\text{filtered}}(k+1) = u_{\text{filtered}}(k) + \alpha_{\text{act}} \left( u_{\text{cmd}}(k) - u_{\text{filtered}}(k) \right)
$$

其中滤波系数：

$$
\alpha_{\text{act}} = \frac{T_s}{\tau + T_s}
$$

| 参数 | 值 |
|------|-----|
| 时间常数 $\tau$ | 0.001 s |
| 采样时间 $T_s$ | 0.02 s |
| 滤波系数 $\alpha_{\text{act}}$ | 0.9524 |

### 5.2 升降舵符号反转

Simulink 模型中存在符号反转（Gain = -1）：

$$
\delta_e^{\text{phys}} = -\delta_e^{\text{filtered}}
$$

---

## 6. 强化学习环境

### 6.1 环境接口

采用 Gymnasium 标准接口：

| 组件 | 维度 | 范围 |
|------|------|------|
| 观测空间 | 11D | $(-\infty, +\infty)$ |
| 动作空间 | 2D | $[-1, 1]$ |

### 6.2 观测空间

11 维观测向量：

$$
\mathbf{o} = \begin{bmatrix} V \\ \alpha \\ \theta \\ q \\ h \\ \delta_{t,\text{last}} \\ \delta_{e,\text{last}} \\ \Delta V \\ \Delta\alpha \\ \Delta\theta \\ \text{cmd} \end{bmatrix}
$$

| 分量 | 说明 |
|------|------|
| $V$ | 当前空速 |
| $\alpha$ | 当前迎角 |
| $\theta$ | 当前俯仰角 |
| $q$ | 当前俯仰角速度 |
| $h$ | 当前高度 |
| $\delta_{t,\text{last}}$ | 上一步油门指令（归一化后） |
| $\delta_{e,\text{last}}$ | 上一步升降舵指令（归一化后） |
| $\Delta V = V_{\text{target}} - V$ | 空速误差 |
| $\Delta\alpha = \alpha_{\text{target}} - \alpha$ | 迎角误差 |
| $\Delta\theta = \theta_{\text{target}} - \theta$ | 俯仰角误差 |
| $\text{cmd}$ | 飞行模式指令 |

### 6.3 动作空间

2 维连续动作空间：

$$
\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \in [-1, 1]^2
$$

动作反归一化后得到物理量：

$$
\begin{bmatrix} \delta_t \\ \delta_e \end{bmatrix} = \mathbf{A}_{\text{norm}}^{-1} \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \mathbf{b}_{\text{act}}
$$

| 物理量 | 范围 |
|--------|------|
| 节气门 $\delta_t$ | [-1.0, 0.0] |
| 升降舵 $\delta_e$ | [-20.0, 30.0] deg |

### 6.4 飞行模式

| cmd | 模式 | 目标状态 |
|-----|------|----------|
| $-1$ | 前飞 | $V=20$ m/s, $\alpha \approx 5°$, $\theta \approx 5°$ |
| $+1$ | 悬停 | $V \approx 0$ m/s, $\alpha \approx 90°$, $\theta \approx 90°$ |

### 6.5 回合重置

**SAC 模式**（99/1 分布）：
- 99% 概率：前飞→悬停回转（cmd=1），初始状态为前飞状态
- 1% 概率：悬停→前飞转换（cmd=-1），初始状态在悬停与前飞之间随机插值

**PPO 模式**（50/50 分布）：
- 50% 概率：前飞→悬停回转
- 50% 概率：悬停→前飞转换

### 6.6 终止条件

| 条件 | 阈值 | 说明 |
|------|------|------|
| 地面坠毁 | $h \leq -20$ m | 高度过低 |
| 超高 | $h \geq 50$ m | 高度过高 |
| 迎角过小 | $\alpha \leq -50°$ | 失速 |
| 迎角过大 | $\alpha \geq 120°$ | 失速 |
| 俯仰角过小 | $\theta \leq -50°$ | 姿态异常 |
| 俯仰角过大 | $\theta \geq 120°$ | 姿态异常 |
| 俯仰角过大 | $\theta \geq 25°$ | 前飞模式限制 |
| 成功 | $I_{\text{success}} \geq 1$ | 达到目标 |
| 超时 | $t \geq 20$ s | 时间限制 |

---

## 7. 奖励函数

奖励函数逆向工程自 `RL/LonEnv.slx`，包含 5 个分量。

### 7.1 非线性整形函数

定义整形函数：

$$
f(x) = \frac{1}{\sqrt{0.1 x^2 + 1}}
$$

性质：
- $f(0) = 1$（峰值）
- $f(x)$ 随 $|x|$ 增大而单调递减
- $f(x) > 0, \quad \forall x \in \mathbb{R}$

### 7.2 成功积分

定义加权状态误差：

$$
\mathbf{e} = \begin{bmatrix} 0.5 \Delta V \\ 0.5 \times 57.3 \times \Delta\alpha \\ 0.5 \times 57.3 \times \Delta\theta \end{bmatrix}
$$

其中 $57.3 = 180/\pi$ 为弧度→度转换因子。

误差范数：

$$
\|\mathbf{e}\|_2 = \sqrt{(0.5 \Delta V)^2 + (0.5 \times 57.3 \times \Delta\alpha)^2 + (0.5 \times 57.3 \times \Delta\theta)^2}
$$

成功指示器：

$$
I_{\text{success}}(t) = \begin{cases} 1 & \text{if } \|\mathbf{e}\|_2 < 2.0 \\ 0 & \text{otherwise} \end{cases}
$$

成功积分（累加并限幅）：

$$
I_{\text{success}}^{\text{int}}(t+1) = \min\left(I_{\text{success}}^{\text{int}}(t) + I_{\text{success}}(t) \cdot T_s, \; 1.0\right)
$$

### 7.3 奖励分量

**成功奖励**：

$$
r_{\text{success}} = 1000 \cdot I_{\text{success}}^{\text{int}}
$$

**空速跟踪奖励**：

$$
r_V = f\left(5 \cdot |V - V_{\text{target}}|\right)
$$

**俯仰角跟踪奖励**：

$$
r_\theta = f\left(|\theta_{\text{deg}} - \theta_{\text{target,deg}}|\right)
$$

**迎角跟踪奖励**：

$$
r_\alpha = f\left(|\alpha_{\text{deg}} - \alpha_{\text{target,deg}}|\right)
$$

**高度变化率惩罚**：

$$
r_{\dot{h}} = 0.05 \cdot w^2
$$

其中 $w$ 为机体 z 轴速度，近似于 $dh/dt$。

**时间惩罚**：

$$
r_{\text{time}} = -0.05 \cdot e^{0.1 t}
$$

### 7.4 总奖励

$$
R(t) = r_{\text{time}} + r_{\text{success}} + r_V + r_\theta + r_\alpha + r_{\dot{h}}
$$

展开：

$$
R(t) = -0.05 e^{0.1 t} + 1000 I_{\text{success}}^{\text{int}} + \frac{1}{\sqrt{0.1(5|ΔV|)^2 + 1}} + \frac{1}{\sqrt{0.1(Δθ°)^2 + 1}} + \frac{1}{\sqrt{0.1(Δα°)^2 + 1}} + 0.05 w^2
$$

### 7.5 奖励特性分析

| 分量 | 范围 | 作用 |
|------|------|------|
| $r_{\text{time}}$ | $(-\infty, -0.05]$ | 指数增长的时间惩罚，鼓励快速完成 |
| $r_{\text{success}}$ | $[0, 1000]$ | 主导奖励，到达目标后持续累积 |
| $r_V$ | $(0, 1]$ | 空速跟踪，误差越小奖励越高 |
| $r_\theta$ | $(0, 1]$ | 俯仰角跟踪 |
| $r_\alpha$ | $(0, 1]$ | 迎角跟踪 |
| $r_{\dot{h}}$ | $[0, +\infty)$ | 惩罚垂直速度，鼓励平稳飞行 |

---

## 8. 归一化方法

### 8.1 观测归一化

观测向量通过矩阵乘法归一化：

$$
\mathbf{o}_{\text{norm}} = \mathbf{N}_s \cdot \mathbf{o}_{\text{raw}}
$$

其中 $\mathbf{N}_s$ 为 11×11 对角矩阵：

$$
\mathbf{N}_s = \text{diag}([0.1, 0.05, 0.05, 0.1, 0.5, 1.0, 0.05, 0.2, 0.1, 0.1, 1.0])
$$

| 分量 | 缩放因子 | 说明 |
|------|----------|------|
| $V$ | 0.1 | 空速 ≈ 20 m/s → 2.0 |
| $\alpha$ | 0.05 | 迎角 ≈ 1.57 rad → 0.079 |
| $\theta$ | 0.05 | 俯仰角 ≈ 1.57 rad → 0.079 |
| $q$ | 0.1 | 俯仰角速度 |
| $h$ | 0.5 | 高度 ≈ 10 m → 5.0 |
| $\delta_{t,\text{last}}$ | 1.0 | 归一化动作 |
| $\delta_{e,\text{last}}$ | 0.05 | 归一化动作 |
| $\Delta V$ | 0.2 | 空速误差 |
| $\Delta\alpha$ | 0.1 | 迎角误差 |
| $\Delta\theta$ | 0.1 | 俯仰角误差 |
| $\text{cmd}$ | 1.0 | 模式指令 |

### 8.2 动作反归一化

将智能体输出 $[-1, 1]$ 转换为物理单位：

$$
\begin{bmatrix} \delta_t \\ \delta_e \end{bmatrix} = \begin{bmatrix} 1/2 & 0 \\ 0 & 1/0.04 \end{bmatrix} \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} -0.5 \\ 5.0 \end{bmatrix}
$$

$$
\delta_t = 0.5 a_1 - 0.5
$$

$$
\delta_e = 25 a_2 + 5.0
$$

| 动作 | 智能体输出 | 物理范围 |
|------|-----------|----------|
| 节气门 $\delta_t$ | $[-1, 1]$ | $[-1.0, 0.0]$ |
| 升降舵 $\delta_e$ | $[-1, 1]$ | $[-20.0, 30.0]$ deg |

---

## 9. 强化学习算法

### 9.1 SAC（Soft Actor-Critic）

SAC 是一种基于最大熵框架的离策略（off-policy）Actor-Critic 算法。

**目标函数**：

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t) \sim \rho_\pi} \left[ r(\mathbf{s}_t, \mathbf{a}_t) + \alpha \mathcal{H}(\pi(\cdot | \mathbf{s}_t)) \right]
$$

其中 $\mathcal{H}$ 为策略熵，$\alpha$ 为温度参数。

**策略更新**：

$$
\pi^* = \arg\max_\pi \sum_{t=0}^{T} \mathbb{E} \left[ Q^{\pi}(\mathbf{s}_t, \mathbf{a}_t) - \alpha \log \pi(\mathbf{a}_t | \mathbf{s}_t) \right]
$$

**Q 函数更新**（最小化 Bellman 残差）：

$$
L_Q(\theta) = \mathbb{E} \left[ \left( Q_\theta(\mathbf{s}, \mathbf{a}) - y \right)^2 \right]
$$

$$
y = r + \gamma \left( Q_{\bar{\theta}}(\mathbf{s}', \tilde{\mathbf{a}}') - \alpha \log \pi_\phi(\tilde{\mathbf{a}}' | \mathbf{s}') \right), \quad \tilde{\mathbf{a}}' \sim \pi_\phi(\cdot | \mathbf{s}')
$$

**温度自动调节**：

$$
\alpha^* = \arg\min_\alpha \mathbb{E}_{\mathbf{a} \sim \pi^*} \left[ -\alpha \log \pi^*(\mathbf{a} | \mathbf{s}) - \alpha \bar{\mathcal{H}} \right]
$$

**超参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| $\gamma$ | 0.99 | 折扣因子 |
| $\tau$ | 0.001 | 软更新系数 |
| 学习率 | 0.001 | Adam 优化器 |
| 经验池大小 | 100,000 | Replay buffer |
| 批大小 | 512 | Mini-batch |
| 预热步数 | 256 | 训练前收集步数 |
| 网络结构 | [256, 256] | 隐藏层维度 |
| 激活函数 | ReLU | — |
| 总步数 | 40,000,000 | ≈ 40,000 episodes |

### 9.2 PPO（Proximal Policy Optimization）

PPO 是一种基于信赖域的在策略（on-policy）Actor-Critic 算法。

**目标函数**（Clipped Surrogate）：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中概率比：

$$
r_t(\theta) = \frac{\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)}{\pi_{\theta_{\text{old}}}(\mathbf{a}_t | \mathbf{s}_t)}
$$

**广义优势估计（GAE）**：

$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}
$$

$$
\delta_t = r_t + \gamma V(\mathbf{s}_{t+1}) - V(\mathbf{s}_t)
$$

**总目标函数**：

$$
L(\theta) = L^{CLIP}(\theta) + c_1 L^{VF}(\theta) - c_2 \mathcal{H}(\pi_\theta)
$$

**超参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| $\gamma$ | 0.99 | 折扣因子 |
| $\lambda$ | 0.95 | GAE lambda |
| $\epsilon$ | 0.2 | Clip 范围 |
| $c_2$ | 0.01 | 熵系数 |
| 学习率 | 0.001 | Adam 优化器 |
| 批大小 | 512 | Mini-batch |
| 更新轮数 | 3 | 每次收集数据的训练轮数 |
| 每次收集步数 | 1024 | Rollout 长度 |
| 最大梯度范数 | 1.0 | 梯度裁剪 |
| 值函数系数 $c_1$ | 0.5 | — |
| 网络结构 | [256, 128] | 隐藏层维度 |
| 激活函数 | ReLU | — |
| 总步数 | 40,000,000 | ≈ 40,000 episodes |

### 9.3 算法选择

| 特性 | SAC | PPO |
|------|-----|-----|
| 策略类型 | Off-policy | On-policy |
| 探索方式 | 最大熵（自动） | 熵正则化 |
| 样本效率 | 高 | 低 |
| 训练稳定性 | 中 | 高 |
| 超参数敏感度 | 中 | 低 |
| 适用场景 | 连续控制 | 通用 |

本项目中 SAC 为主要训练算法，PPO 作为对比基线。

---

## 参考文献

1. Haarnoja, T., et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
2. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
3. Schulman, J., et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." ICLR 2016.
4. Original MATLAB/Simulink code: `mdl/Lon_mdl.slx`, `RL/LonEnv.slx`
