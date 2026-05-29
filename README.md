# TailsitterControl

倾转旋翼无人机纵向动力学强化学习控制 / Reinforcement Learning Control for Tailsitter UAV Longitudinal Dynamics

---

## 项目简介 / Overview

**[English](#english)** | **[中文](#中文)**

---

<a id="中文"></a>

## 中文

### 项目简介

本项目使用强化学习（SAC/PPO）训练智能体，控制倾转旋翼无人机在悬停与前飞模式之间的纵向动力学转换。项目从 MATLAB/Simulink 原始代码重构为纯 Python 实现，基于 Stable-Baselines3 + Gymnasium 框架。

### 主要特性

- **完整动力学模型**：5-DOF 纵向动力学，从 `mdl/Lon_mdl.slx` 逆向工程获得
- **精确奖励函数**：5 分量奖励函数，从 `RL/LonEnv.slx` 逆向工程获得
- **双算法支持**：支持 SAC 和 PPO 两种强化学习算法
- **模块化设计**：动力学、奖励、归一化、重置逻辑独立模块
- **配平分析**：稳态配平求解、过渡走廊扫描、线性化与特征值分析
- **轨迹优化**：CasADi + IPOPT 直接配点法求解最优过渡轨迹
- **微分平坦轨迹优化**：基于微分平坦的 (V, γ) 空间轨迹优化，替代 CasADi 高维方法
- **完整测试**：84 个单元测试覆盖全部核心功能

### 快速开始

#### 环境要求

- Python >= 3.10

#### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd TailsitterControl

# 安装依赖
pip install -e .

# 转换 MATLAB 数据文件（仅需一次）
python scripts/convert_mat_to_npy.py
```

#### 训练

```bash
# 训练 SAC 智能体
python scripts/train.py --algo sac

# 训练 PPO 智能体
python scripts/train.py --algo ppo

# 从检查点继续训练
python scripts/train_continue.py \
    --checkpoint results/DP_sac_40000000/agent \
    --algo sac \
    --timesteps 20000000
```

#### 评估

```bash
# 评估训练好的智能体
python scripts/evaluate.py \
    --checkpoint results/DP_sac_40000000/agent \
    --algo sac
```

评估结果保存在 `results/evaluation/` 目录下，包含前飞转换和回转转换的响应曲线。

#### 配平分析

```bash
# 单点配平
python scripts/trim.py point --V 15 --theta 10

# 过渡走廊扫描
python scripts/trim.py corridor --v-step 1 --theta-step 5

# 走廊扫描 + 线性化分析
python scripts/trim.py corridor --v-step 2 --theta-step 10 --linearize
```

#### 轨迹优化

```bash
# 前飞 → 悬停
python scripts/trajectory_optimize.py --direction forward2hover

# 两阶段求解（粗网格 → 精网格热启动）
python scripts/trajectory_optimize.py --direction forward2hover --two-stage

# L-BFGS Hessian + 单调屏障策略
python scripts/trajectory_optimize.py --direction forward2hover --hessian limited-memory --mu-strategy monotone
```

#### 微分平坦轨迹优化

```bash
# DF 走廊扫描
python scripts/df_trajectory.py df-trim

# DF 轨迹优化（悬停 → 前飞）
python scripts/df_trajectory.py df-optimize --direction hover2forward

# DF 轨迹对比（优化方法 vs 基线方法）
python scripts/df_trajectory.py df-compare --direction hover2forward

# DF 轨迹跟踪仿真 + 误差分析
python scripts/df_trajectory.py df-simulate --direction hover2forward
```

#### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行单个测试
python -m pytest tests/test_env.py::test_observation_space_shape -v
```

### 项目结构

```
TailsitterControl/
├── config/                    # YAML 配置文件
│   ├── physical.yaml          # 物理参数（质量、气动、螺旋桨）
│   ├── normalization.yaml     # 状态/动作归一化矩阵
│   ├── sac_hyperparams.yaml   # SAC 超参数
│   └── ppo_hyperparams.yaml   # PPO 超参数
├── data/
│   ├── raw/                   # 原始 .mat/.xlsx 文件
│   └── processed/             # 转换后的 .npy 文件
├── docs/                      # 补充文档
│   ├── theory.md              # 理论基础（动力学、RL、奖励函数）
│   ├── architecture.md        # 架构设计说明
│   ├── user_guide.md          # 用户指南
│   └── development.md         # 开发指南
├── scripts/                   # 训练、评估和分析脚本
│   ├── convert_mat_to_npy.py  # 数据转换
│   ├── train.py               # 训练入口
│   ├── train_continue.py      # 断点续训
│   ├── evaluate.py            # 评估脚本
│   ├── trim.py                # 配平分析与过渡走廊
│   ├── trajectory_optimize.py # CasADi 轨迹优化
│   └── df_trajectory.py       # 微分平坦轨迹优化
├── src/tailsitter/            # 核心 Python 包
│   ├── config.py              # 配置加载器
│   ├── dynamics.py            # 动力学模型
│   ├── env.py                 # Gymnasium 环境
│   ├── linearization.py       # 线性化与特征值分析
│   ├── normalization.py       # 状态/动作归一化
│   ├── plotting.py            # 可视化
│   ├── reset.py               # 回合重置逻辑
│   ├── reward.py              # 奖励函数
│   ├── trajectory_optimization.py # CasADi 轨迹优化
│   ├── trajectory_plotting.py # 轨迹结果可视化
│   ├── trim.py                # 配平求解与走廊扫描
│   ├── trim_plotting.py       # 走廊热力图可视化
│   ├── differential_flatness.py # 微分平坦传递与俯仰角速率裕度
│   ├── df_trim.py             # 微分平坦走廊扫描
│   ├── df_trajectory_optimization.py # 微分平坦轨迹优化器
│   ├── baseline_trajectory.py # 基线走廊路径优化
│   ├── df_trajectory_plotting.py # 微分平坦轨迹可视化
│   └── df_trajectory_simulation.py # 微分平坦轨迹跟踪仿真 + 误差分析
└── tests/                     # 单元测试
```

### 技术细节

#### 状态空间

- **飞行器状态 (7D)**: `[u, w, V, α, θ, q, h]`
- **观测空间 (11D)**: `[V, α, θ, q, h, dt_last, de_last, dV, dα, dθ, cmd]`

#### 动作空间

`[dt, de]` ∈ [-1, 1]，通过归一化矩阵转换为物理单位

#### 飞行模式

| cmd | 模式 | 目标状态 |
|-----|------|----------|
| -1 | 前飞 | V=20 m/s, α≈5°, θ≈5° |
| 1 | 悬停 | V≈0 m/s, α≈90°, θ≈90° |

### 文档

- [理论文档](docs/theory.md) — 动力学模型、强化学习方法、奖励函数公式推导
- [架构设计](docs/architecture.md) — 技术细节和设计决策
- [用户指南](docs/user_guide.md) — 使用说明
- [开发指南](docs/development.md) — 开发流程和里程碑
- [轨迹优化计划](docs/trajectory_optimization_plan.md) — 收敛改进计划与进展
- [更新日志](CHANGELOG.md) — 版本历史

### 许可证

[待定]

---

<a id="english"></a>

## English

### Overview

This project uses reinforcement learning (SAC/PPO) to train agents that control a tailsitter UAV's longitudinal dynamics during flight mode transitions between hover and forward flight. Refactored from MATLAB/Simulink to pure Python using Stable-Baselines3 + Gymnasium.

### Features

- **Complete Dynamics Model**: 5-DOF longitudinal dynamics reverse-engineered from `mdl/Lon_mdl.slx`
- **Precise Reward Function**: 5-component reward function reverse-engineered from `RL/LonEnv.slx`
- **Dual Algorithm Support**: SAC and PPO reinforcement learning algorithms
- **Modular Design**: Separate modules for dynamics, reward, normalization, and reset logic
- **Trim Analysis**: Steady-state trim solvers, transition corridor sweep, linearization and eigenvalue analysis
- **Trajectory Optimization**: CasADi + IPOPT direct collocation for optimal transition trajectories
- **DF Trajectory Optimization**: Differential-flatness-based trajectory optimization in (V, γ) space, alternative to CasADi high-dimensional approach
- **Comprehensive Testing**: 84 unit tests covering all core functionality

### Quick Start

#### Requirements

- Python >= 3.10

#### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TailsitterControl

# Install dependencies
pip install -e .

# Convert MATLAB data files (one-time)
python scripts/convert_mat_to_npy.py
```

#### Training

```bash
# Train SAC agent
python scripts/train.py --algo sac

# Train PPO agent
python scripts/train.py --algo ppo

# Continue training from checkpoint
python scripts/train_continue.py \
    --checkpoint results/DP_sac_40000000/agent \
    --algo sac \
    --timesteps 20000000
```

#### Evaluation

```bash
# Evaluate trained agent
python scripts/evaluate.py \
    --checkpoint results/DP_sac_40000000/agent \
    --algo sac
```

Results are saved to `results/evaluation/` as PNG plots showing forward and back transition responses.

#### Trim Analysis

```bash
# Single point trim
python scripts/trim.py point --V 15 --theta 10

# Transition corridor sweep
python scripts/trim.py corridor --v-step 1 --theta-step 5

# Corridor with linearization analysis
python scripts/trim.py corridor --v-step 2 --theta-step 10 --linearize
```

#### Trajectory Optimization

```bash
# Forward flight → hover
python scripts/trajectory_optimize.py --direction forward2hover

# Two-stage solve (coarse → fine, warm-started)
python scripts/trajectory_optimize.py --direction forward2hover --two-stage

# L-BFGS Hessian with monotone barrier strategy
python scripts/trajectory_optimize.py --direction forward2hover --hessian limited-memory --mu-strategy monotone
```

#### DF-Based Trajectory Optimization

```bash
# DF corridor sweep
python scripts/df_trajectory.py df-trim

# DF trajectory optimization (hover → forward flight)
python scripts/df_trajectory.py df-optimize --direction hover2forward

# DF trajectory comparison (proposed vs baseline)
python scripts/df_trajectory.py df-compare --direction hover2forward

# DF trajectory tracking simulation + error analysis
python scripts/df_trajectory.py df-simulate --direction hover2forward
```

#### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run single test
python -m pytest tests/test_env.py::test_observation_space_shape -v
```

### Project Structure

```
TailsitterControl/
├── config/                    # YAML configuration files
│   ├── physical.yaml          # Physical parameters (mass, aero, prop)
│   ├── normalization.yaml     # State/action normalization matrices
│   ├── sac_hyperparams.yaml   # SAC hyperparameters
│   └── ppo_hyperparams.yaml   # PPO hyperparameters
├── data/
│   ├── raw/                   # Original .mat/.xlsx files
│   └── processed/             # Converted .npy files
├── docs/                      # Supplementary documentation
│   ├── theory.md              # Theoretical foundations (dynamics, RL, reward)
│   ├── architecture.md        # Architecture and design decisions
│   ├── user_guide.md          # Usage instructions
│   └── development.md         # Development workflow
├── scripts/                   # Training, evaluation, and analysis scripts
│   ├── convert_mat_to_npy.py  # Data conversion
│   ├── train.py               # Training entry point
│   ├── train_continue.py      # Continue from checkpoint
│   ├── evaluate.py            # Evaluation script
│   ├── trim.py                # Trim analysis and transition corridor
│   ├── trajectory_optimize.py # CasADi trajectory optimization
│   └── df_trajectory.py       # DF-based trajectory optimization
├── src/tailsitter/            # Core Python package
│   ├── config.py              # Configuration loaders
│   ├── dynamics.py            # Dynamics model
│   ├── env.py                 # Gymnasium environment
│   ├── linearization.py       # Linearization and eigenvalue analysis
│   ├── normalization.py       # State/action normalization
│   ├── plotting.py            # Visualization
│   ├── reset.py               # Episode reset logic
│   ├── reward.py              # Reward function
│   ├── trajectory_optimization.py # CasADi trajectory optimization
│   ├── trajectory_plotting.py # Trajectory result visualization
│   ├── trim.py                # Trim solvers and corridor sweep
│   ├── trim_plotting.py       # Corridor heatmap visualization
│   ├── differential_flatness.py # DF transfer and pitch rate margin
│   ├── df_trim.py             # DF-based corridor sweep
│   ├── df_trajectory_optimization.py # DF trajectory optimizer
│   ├── baseline_trajectory.py # Baseline corridor path optimization
│   ├── df_trajectory_plotting.py # DF trajectory visualization
│   └── df_trajectory_simulation.py # DF trajectory tracking simulation + error analysis
└── tests/                     # Unit tests
```

### Technical Details

#### State Space

- **Plant State (7D)**: `[u, w, V, α, θ, q, h]`
- **Observation (11D)**: `[V, α, θ, q, h, dt_last, de_last, dV, dα, dθ, cmd]`

#### Action Space

`[dt, de]` ∈ [-1, 1], denormalized to physical units via normalization matrix

#### Flight Modes

| cmd | Mode | Target State |
|-----|------|--------------|
| -1 | Forward flight | V=20 m/s, α≈5°, θ≈5° |
| 1 | Hover | V≈0 m/s, α≈90°, θ≈90° |

### Documentation

- [Theory](docs/theory.md) — Dynamics model, RL methods, reward function derivations
- [Architecture](docs/architecture.md) — Technical details and design decisions
- [User Guide](docs/user_guide.md) — Usage instructions
- [Development Guide](docs/development.md) — Development workflow and milestones
- [Trajectory Optimization Plan](docs/trajectory_optimization_plan.md) — Convergence improvement plan and progress
- [Changelog](CHANGELOG.md) — Version history

### License

[To be determined]
