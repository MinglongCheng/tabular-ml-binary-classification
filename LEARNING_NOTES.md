# Python 二分类机器学习笔记

## 目录

- [实践流程](#workflow)
- [一、数据预处理](#preprocessing)
  - [1. 数据预处理的目的及重要性](#preprocessing-purpose)
  - [2. 数据理解和探索](#data-exploration)
  - [3. 数据清洗](#data-cleaning)
  - [4. 数据拆分](#data-splitting)
  - [5. 特征工程](#feature-engineering)
- [二、预测模型构建及调参](#modeling)
  - [1. 构建模型总览与超参数调优](#model-overview)
  - [2. Logistic 回归模型](#logistic-regression)
  - [3. 决策树模型](#decision-tree)
  - [4. 随机森林模型](#random-forest)
  - [5. XGBoost 模型](#xgboost)
  - [6. LightGBM 模型](#lightgbm)
  - [7. 支持向量机模型](#svm)
  - [8. 人工神经网络模型](#ann)
- [三、模型预测效果评价](#evaluation)
  - [1. 预测效果评价方式](#evaluation-overview)
  - [2. 混淆矩阵及分类指标](#confusion-matrix)
  - [3. ROC 曲线与 AUC](#roc-auc)
  - [4. 校准曲线](#calibration)
  - [5. 决策曲线分析](#dca)
  - [6. 预测效果评价总结](#evaluation-summary)
- [四、机器学习可解释性](#interpretability)
  - [1. 机器学习可解释性的由来](#interpretability-purpose)
  - [2. 机器学习可解释性方法](#interpretability-methods)
  - [3. SHAP 值及其计算](#shap-values)
  - [4. SHAP 特征重要性条形图](#shap-bar)
  - [5. SHAP 特征重要性蜂群图](#shap-beeswarm)
  - [6. SHAP 依赖性图](#shap-dependence)
  - [7. SHAP 瀑布图](#shap-waterfall)
  - [8. SHAP 力图](#shap-force)
  - [9. SHAP 图形总结](#shap-summary)

---

<a id="workflow"></a>
## 实践流程

二分类机器学习模型及其可解释性实践可以分为四个主要阶段：

**数据准备 → 构建模型 → 评估模型 → SHAP 可解释性分析。**

| 阶段 | 主要内容 | 需要回答的问题 |
| --- | --- | --- |
| 数据准备 | 数据理解和探索、数据清洗、数据分割、数据转换 | 数据包含什么信息？是否存在异常值和缺失值？哪些变量是特征，哪个变量是结局？ |
| 构建模型 | 选择算法、训练模型、优化调参 | 使用什么模型？如何训练？如何选择超参数？ |
| 评估模型 | 混淆矩阵、ROC 曲线、校准曲线、决策曲线 | 分类是否准确？区分能力如何？预测概率是否可靠？在不同决策阈值下有何表现？ |
| SHAP | 条形图、蜂群图、依赖性图、瀑布图、力图 | 哪些特征对模型输出重要？它们怎样影响整体和个体预测？ |

---

<a id="preprocessing"></a>
## 一、数据预处理

<a id="preprocessing-purpose"></a>
### 1. 数据预处理的目的及重要性

数据预处理的主要目的是通过清洗、转换和准备数据，从而能够有效地输入到机器学习算法中。

数据预处理是机器学习模型开发中至关重要的一步，将直接影响模型的预测效果。

数据预处理包括以下主要步骤：

**数据理解和探索 → 数据清洗 → 数据拆分 → 特征工程。**

<a id="data-exploration"></a>
### 2. 数据理解和探索

数据理解和探索主要包括：

1. 查看数据集中每个变量的类型，例如数值型、字符型等，以及它们的意义。
2. 识别哪些是特征变量（$`X`$），哪些是目标变量（$`Y`$）。
3. 通过直方图、箱线图等可视化手段，了解数据中变量的分布特点，是否存在异常值、缺失值，从而进行后续处理。
4. 识别哪些变量可能需要进行特征工程，例如独热编码、标准化等。

#### 2.1 常用数据查看操作

```python
print(data.head())  # 查看前五行数据
data.info()  # 查看数据结构及非缺失数量
data['diabetes'].value_counts()  # 查看结局变量的频数
data['diabetes'].value_counts(normalize=True)  # 查看结局变量的频率
data.isnull().sum()  # 查看各变量的缺失数
```

| 操作 | 作用 |
| --- | --- |
| `head()` | 查看数据集前几行，初步了解字段与数值形式 |
| `info()` | 查看变量名、数据类型及非缺失值数量 |
| `value_counts()` | 统计各类别出现的次数 |
| `value_counts(normalize=True)` | 统计各类别所占比例 |
| `isnull().sum()` | 按列统计缺失值数量 |

#### 2.2 直方图与箱线图

直方图用于查看连续变量的分布；箱线图用于观察中位数、四分位数及偏离主体分布的观测。

![Glucose 的直方图与箱线图](outputs/figures/eda/glucose_histogram_boxplot.png)

<a id="data-cleaning"></a>
### 3. 数据清洗

#### 3.1 处理异常值（Outliers）

异常值是偏离其他数据点的极端值，可能会影响模型的训练。

**处理异常值需要先分析具体原因，不能一概而论。**

常用处理方法：

| 方法 | 基本思路 |
| --- | --- |
| IQR 方法 | 通过计算四分位数间距（Interquartile Range，IQR）识别异常值 |
| Z-score 方法 | 通过计算每个数据点的 Z-score，识别与均值差异过大的数据点 |
| 基于模型的方法 | 使用聚类、决策树等方法识别和去除异常值 |

**IQR 方法的计算**

设第一四分位数为 $`Q_1`$，第三四分位数为 $`Q_3`$，则：

$$
\mathrm{IQR}=Q_3-Q_1
$$

采用 1.5 倍 IQR 规则，上下界为：

$$
\mathrm{Lower}=Q_1-1.5\,\mathrm{IQR},\qquad
\mathrm{Upper}=Q_3+1.5\,\mathrm{IQR}
$$

低于下界或高于上界的观测被标记为需要检查的异常观测。

```python
Q1 = data['glucose'].quantile(0.25)  # 第一四分位数
Q3 = data['glucose'].quantile(0.75)  # 第三四分位数
IQR = Q3 - Q1  # 四分位间距
lower_bound = Q1 - 1.5 * IQR  # 下界
upper_bound = Q3 + 1.5 * IQR  # 上界
outliers_index = data[(data['glucose'] < lower_bound) |
                      (data['glucose'] > upper_bound)].index  # 异常观测的索引
data.loc[outliers_index, :]  # 查看异常观测
```

#### 3.2 处理缺失值（Missing Values）

数据缺失问题是数据分析中的常见问题。

常用处理方法包括：

| 方法 | 说明 |
| --- | --- |
| 删除缺失值 | 适用于缺失值比例很低的情况下 |
| 填充缺失值 | 使用均值、中位数或众数填充缺失值，或者使用其他列的相关性进行填充，例如使用回归填充缺失值 |
| 插值法 | 使用插值方法估算缺失值 |

具体采用哪一种方法，需要结合数据结构和缺失原因，而不是只比较缺失值数量。

#### 3.3 MissForest 随机森林插补

**基本原理**

MissForest 是一种基于随机森林的数据插补方法。它通过利用多棵决策树来预测缺失值，并不断迭代更新预测结果，直到收敛为止。

**填补流程**

1. 对于含有缺失值的数据集，首先对每个含有缺失值的变量，将缺失值对应的变量视为响应变量，其余变量视为特征变量，构建预测模型。
2. 利用已有的非缺失值作为训练数据，预测缺失值。
3. 将预测得到的值作为缺失值的估计，并更新数据集，不断重复上述训练和预测步骤，直到达到收敛标准或者预设的迭代次数。

**优点**

- **非线性关系适应性：** 基于树模型的方法能够捕捉非线性关系，对非线性关系数据具有较好的适应性。
- **对异常值的鲁棒性：** 随机森林模型对异常值具有一定鲁棒性，在包含异常值的数据中可能表现较好。

**缺点**

- **计算时间较长：** 对于较大的数据集，需要构建多棵树并进行迭代更新，因此计算时间可能比较耗时。
- **对缺失模式的敏感性：** 不同缺失模式可能影响插补结果，需谨慎处理数据中的缺失情况。

**Python 函数与常用参数**

```python
from missforest import MissForest

imputer = MissForest()  # 创建插补器
X_train_imputed = imputer.fit_transform(X_train)  # 使用训练集拟合并填补
X_val_imputed = imputer.transform(X_val)  # 使用已拟合的插补器填补验证集
X_test_imputed = imputer.transform(X_test)  # 使用已拟合的插补器填补测试集
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `clf` | `lgbm_clf` | 用于填补分类变量的模型，该实现默认使用 LightGBM 分类器 |
| `rgr` | `lgbm_rgr` | 用于填补数值变量的模型，该实现默认使用 LightGBM 回归器 |
| `categorical` | `None` | 指定哪些变量为分类变量，按相应实现要求提供列名或列索引 |
| `initial_guess` | `"median"` | 缺失值的初始填补方法，可设置为 `"mean"` 或 `"median"` |
| `max_iter` | `5` | 最大迭代次数，填补过程在此轮数内完成或提前终止 |
| `early_stopping` | `True` | 是否启用早停，根据停止准则提前结束迭代 |
| `verbose` | `2` | 控制运行信息的显示程度 |

<a id="data-splitting"></a>
### 4. 数据拆分（Data Splitting）

#### 4.1 训练集、验证集与测试集

| 数据集 | 作用 |
| --- | --- |
| 训练集（Training Set） | 训练模型的部分数据，模型通过这些数据学习规律 |
| 验证集（Validation Set） | 在训练过程中评估模型性能的数据，用于调整模型超参数 |
| 测试集（Test Set） | 最终评估模型性能的数据，不用于训练或超参数选择 |

常见的数据划分比例包括：

| 比例 | 含义 |
| --- | --- |
| 70%–30% | 70% 用于训练集，30% 用于测试集，是较常见的拆分比例 |
| 80%–20% | 80% 用于训练集，20% 用于测试集 |
| 60%–20%–20% | 60% 用于训练集，20% 用于验证集，20% 用于测试集 |

当先划出测试集、再从其余样本中划分训练集和验证集时，可以将训练集与验证集的合并部分称为开发集（Development Set），以避免两个不同范围的数据都被称作“训练集”而混淆。

#### 4.2 Python 函数与常用参数

`train_test_split()` 用于将数据拆分为两个子集。

```python
from sklearn.model_selection import train_test_split

development_data, test_data = train_test_split(
    data, test_size=0.3,
    stratify=data['diabetes'], random_state=2025
)  # 划分开发集和测试集

train_data, val_data = train_test_split(
    development_data, test_size=0.3,
    stratify=development_data['diabetes'], random_state=123
)  # 从开发集中划分训练集和验证集
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `test_size` | `None` | 测试部分占比；可设为浮点数，也可设为整数指定样本数 |
| `train_size` | `None` | 训练部分占比；未指定时通常由剩余样本确定 |
| `stratify` | `None` | 是否按类别分层抽样；二分类任务中可传入结局标签，尽量保持类别比例 |
| `random_state` | `None` | 设置随机种子，便于复现划分结果 |
| `shuffle` | `True` | 是否在拆分前打乱数据 |

#### 4.3 数据拆分与后续处理的关系

**检查异常值等之后，先拆分数据集，后仅在训练集中拟合插补、归一化、标准化等规则并进行特征工程，以防数据泄露。**

训练集用于学习模型与数据处理规则，验证集用于模型选择，测试集用于最终独立评价。

`fit` 表示学习规则，`transform` 表示应用已有规则。

**训练集：`fit_transform()`；验证集和测试集：`transform()`。**

<a id="feature-engineering"></a>
### 5. 特征工程（Feature Engineering）

#### 5.1 数据标准化（Standardization）

对数据进行标准化，可以使不同特征间具有可比性，并有助于部分机器学习模型的训练与收敛。

**Z-score 标准化：**

$$
x'=\frac{x-\mu}{\sigma}
$$

其中，$`\mu`$ 为均值，$`\sigma`$ 为标准差。经过中心化和尺度变换后，用于拟合标准化器的数据在相应非恒定特征上的均值为 0、方差为 1。

经过线性标准化后，特征的分布形状不变；当数据特征取值范围或单位差异较大时，可以选择标准化处理。

**Python 函数与常用参数**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # 创建标准化器
X_train_scaled = scaler.fit_transform(X_train)  # 学习训练集均值和标准差，并转换训练集
X_val_scaled = scaler.transform(X_val)  # 使用训练集参数转换验证集
X_test_scaled = scaler.transform(X_test)  # 使用训练集参数转换测试集
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `with_mean` | `True` | 是否进行中心化，即减去均值；设为 `False` 则不中心化 |
| `with_std` | `True` | 是否按标准差缩放；设为 `False` 则不进行此缩放 |
| `copy` | `True` | 是否尽量保留输入数据副本；设为 `False` 不保证所有输入都原地修改 |

#### 5.2 特征选择（Feature Selection）

特征选择的目的是选择对模型预测最有用的特征，减少冗余信息。

常见特征选择方法：

| 方法 | 思路与示例 |
| --- | --- |
| 过滤法（Filter Method） | 基于统计指标选择特征，例如单因素、多因素回归分析 |
| 嵌入法（Embedded Method） | 利用模型训练过程中的特征重要性选择特征，例如 LASSO 回归 |
| 包装法（Wrapper Method） | 使用某种机器学习算法评估特征子集的效果，例如递归特征消除 |

#### 5.3 单因素与多因素 Logistic 回归筛选

一种常用的筛选思路是：先逐个自变量进行单因素 Logistic 回归，再将单因素分析结果显著的变量纳入多因素 Logistic 回归分析。

示例采用的筛选阈值为 $`P<0.05`$。

```python
import statsmodels.api as sm

x = sm.add_constant(train_data[[var]])  # 添加截距
model = sm.Logit(y_train, x).fit(disp=0)  # 单因素 Logistic 回归
coef = model.params[var]  # 回归系数
p_value = model.pvalues[var]  # P 值

X_sig = sm.add_constant(train_data[significant_vars_univ])  # 纳入筛选变量并添加截距
model_multilog = sm.Logit(y_train, X_sig).fit(disp=0)  # 多因素 Logistic 回归
```

| 内容 | 含义 |
| --- | --- |
| `Coefficient` | 回归系数，反映变量在模型中的方向和大小 |
| `Odds Ratio` | 优势比，计算为 $`\exp(\beta)`$ |
| `P-value` | 对相应回归系数进行统计检验得到的 P 值 |
| `sm.add_constant()` | 为设计矩阵添加常数项，即截距 |
| `fit(disp=0)` | 拟合模型，并关闭优化过程的迭代信息显示 |

#### 5.4 LASSO 回归

LASSO（Least Absolute Shrinkage and Selection Operator）是一种广泛用于特征选择和模型正则化的线性回归方法。

核心思想是在进行回归分析时，通过加入 L1 正则化项，使一些回归系数变为零，从而实现特征选择。

**优点**

- **避免过拟合：** 通过正则化限制模型复杂度，减少过拟合风险。
- **提高模型的可解释性：** 通过保留部分变量，突出对模型有用的特征。

**局限**

- LASSO 可能选择不合适的特征；当特征之间存在多重共线性时，对高度相关特征的选择可能存在问题。

**Python 函数与常用参数**

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.001)  # 设置正则化强度
lasso.fit(X_train, y_train)  # 在训练集上拟合
selected_variables = X_train.columns[lasso.coef_ != 0].tolist()  # 提取非零系数对应的变量
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `alpha` | `1.0` | 正则化强度；越大表示惩罚越强，更多系数可能被压缩为 0 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 最大迭代次数 |
| `tol` | `0.0001` | 优化过程的收敛容差 |
| `selection` | `"cyclic"` | 系数更新方式；`"cyclic"` 为循环更新，`"random"` 为随机更新 |
| `random_state` | `None` | 在使用随机更新方式时控制随机过程 |

---

<a id="modeling"></a>
## 二、预测模型构建及调参

<a id="model-overview"></a>
### 1. 构建模型总览与超参数调优

#### 1.1 构建模型总览

| 模型（英文名称） | 简称 | Python 实现 |
| --- | --- | --- |
| 逻辑回归（Logistic Regression） | LR | `statsmodels.api.Logit` |
| 决策树（Decision Tree） | DT | `DecisionTreeClassifier` |
| 随机森林（Random Forest） | RF | `RandomForestClassifier` |
| 极限梯度提升机（eXtreme Gradient Boosting） | XGBoost | `XGBClassifier` |
| 轻量级梯度提升机（Light Gradient Boosting Machine） | LightGBM | `LGBMClassifier` |
| 支持向量机（Support Vector Machine） | SVM | `SVC` |
| 人工神经网络（Artificial Neural Network） | ANN | `MLPClassifier` |

#### 1.2 什么是超参数调优？

超参数调优是指：在训练机器学习模型之前，对那些不会通过模型训练自动学习得到的参数进行设定与优化的过程。

这些参数通常控制模型的结构、学习过程或复杂度。例如，决策树的最大深度 `max_depth`、随机森林的决策树数量 `n_estimators`。

#### 1.3 为什么要进行超参数调优？

**提高模型性能。** 合理的超参数组合可以显著提升模型在验证集或测试集上的预测效果。

**防止过拟合或欠拟合。** 调参有助于控制模型复杂度，增强模型的泛化能力。

**适配不同数据特征。** 不同数据集需要不同的参数设定，调参可以更好地贴合数据规律。

#### 1.4 常见超参数调优方法

常见方法包括网格搜索（Grid Search）、随机搜索（Random Search）、贝叶斯优化（Bayesian Optimization），以及结合交叉验证（Cross Validation）进行评估。

| 方法 | 核心思路 |
| --- | --- |
| 网格搜索 | 预先给出各参数的候选值，遍历候选组合 |
| 随机搜索 | 从预设的参数空间中随机抽取组合 |
| 贝叶斯优化 | 根据已经评价的参数组合，进一步选择有潜力的组合 |
| 交叉验证 | 将数据分成若干折，重复训练与验证；可作为上述搜索方法的评价方式 |

#### 1.5 模型训练、选择与最终评价

常用流程是**训练集拟合、验证集比较、测试集最终评价**。本例对六种非 Logistic 模型进行网格搜索，Logistic 模型不进行超参数网格搜索。

以下片段只演示网格搜索的基本逻辑：

```python
best_auc = -np.inf  # 记录最高验证集 AUC
best_model = None  # 记录最佳模型

for max_depth in [3, 5, 10, None]:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=123)
    model.fit(X_train, y_train)  # 使用训练集拟合
    y_val_pred_prob = model.predict_proba(X_val)[:, 1]  # 验证集预测概率
    auc_value = roc_auc_score(y_val, y_val_pred_prob)
    if auc_value > best_auc:
        best_auc = auc_value
        best_model = model
```

<a id="logistic-regression"></a>
### 2. Logistic 回归模型（Logistic Regression，LR）

#### 2.1 基本原理

逻辑回归是经典的统计分类算法，非常常见于二分类问题。它通过建立一个输入变量与输出类别之间的关系模型，预测类别标签。

尽管名为“回归”，它实际上常用作分类算法，主要用于预测分类概率。

在线性预测值的基础上，构建 Sigmoid 函数（逻辑函数），使输出值在 0 和 1 之间，从而适合处理概率问题：

$$
P(Y=1\mid X)=\frac{1}{1+e^{-(\beta_0+\beta_1X_1+\cdots+\beta_pX_p)}}
$$

其中：

| 符号 | 含义 |
| --- | --- |
| $`P(Y=1\mid X)`$ | 给定特征后，预测属于类别 1 的概率 |
| $`\beta_0`$ | 截距 |
| $`\beta_1,\ldots,\beta_p`$ | 回归系数 |
| $`X_1,\ldots,X_p`$ | 输入特征 |

输出概率通常还需根据阈值转为分类结果，例如以 0.5 为阈值：预测概率不低于 0.5 时判为 1，否则判为 0。

#### 2.2 优点

- **简单易懂：** 模型形式直观，易于解释自变量与因变量之间的关系。
- **容易实现且计算效率高：** 大多数统计软件均能实现，模型计算速度快，适用于大规模数据集。
- **假设检验：** 可进行模型系数的假设检验和置信区间估计，提供统计显著性的信息。
- **可解释性强：** 可通过模型系数得到的 OR 值解释变量的影响方向与大小。

#### 2.3 缺点

- **对关系形式有假设：** 模型的线性假设可能不适合一些数据关系。
- **敏感性：** 对异常值和多重共线性敏感，可能影响模型稳定性。
- **需要人为指定模型具体形式：** 对复杂的非线性关系，可能需要加入交互项、非线性项等，模型形式的选择较困难。

#### 2.4 Python 函数与常用参数

```python
import statsmodels.api as sm

X_train_const = sm.add_constant(X_train)  # 添加常数项（截距）
logist_model = sm.Logit(y_train, X_train_const).fit(disp=0)  # 拟合模型
logist_model.summary()  # 查看模型结果

X_test_const = sm.add_constant(X_test, has_constant='add')  # 为预测数据添加截距
y_pred_prob = logist_model.predict(X_test_const)  # 预测类别 1 的概率
y_pred = (y_pred_prob >= 0.5).astype(int)  # 按阈值转换为分类结果
```

| 参数或方法 | 示例值／用法 | 说明 |
| --- | --- | --- |
| `endog` | 必须提供 | 因变量，二分类结局通常为 0／1 |
| `exog` | 必须提供 | 自变量设计矩阵；使用矩阵接口时需自行添加截距 |
| `fit(disp=...)` | `disp=1`；示例使用 `0` | 是否显示拟合过程；`0` 表示关闭迭代信息 |
| `fit(maxiter=...)` | 按需要设置 | 最大迭代次数 |
| `summary()` | 无需传入数据 | 查看回归结果汇总 |
| `predict(exog)` | 传入与训练设定一致的设计矩阵 | 输出预测概率 |

<a id="decision-tree"></a>
### 3. 决策树模型（Decision Tree，DT）

#### 3.1 基本原理

决策树是一种基于树形结构的监督学习算法，用于分类和回归任务，也是一些机器学习集成模型，例如随机森林、XGBoost 的基础组成单元。

决策树将数据逐步划分成子集，从特征中学习简单的决策规则，最终按这些规则预测目标值。

| 组成部分 | 含义 |
| --- | --- |
| 根节点 | 起始节点，代表整个数据集 |
| 决策／内部节点 | 每个节点代表一条分裂规则，依据规则将数据划分为不同部分 |
| 叶节点 | 代表最终的预测结果 |

例如，可以按照“月薪是否达到 50,000”把数据分成“达到”和“未达到”两部分。连续进行这种分裂，就形成了树形规则。

#### 3.2 优点

- **模型结构简单直观，预测结果易理解：** 决策树可以通过图形表现决策规则，便于不熟悉算法的人理解。
- **不需要太多数据预处理：** 树模型通常不需要为特征比例或尺度进行归一化。
- **无需人为指定具体函数形式：** 能够学习自变量与结局之间的非线性关系。

#### 3.3 缺点

- **容易过拟合：** 特征较多、剪枝不足时，树可能变得过深，导致泛化能力下降。
- **模型结构不稳定：** 对异常值和噪声数据敏感，数据的小幅变化可能引起树结构较大变化。
- **可能偏好取值较多的特征：** 部分基于分裂准则的树方法可能偏向可提供更多候选划分的特征。

#### 3.4 Python 函数与常用参数

```python
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(random_state=123)  # 创建决策树分类模型
tree_model.fit(X_train, y_train)  # 训练模型
y_pred_prob = tree_model.predict_proba(X_test)[:, 1]  # 预测类别 1 的概率
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `criterion` | `"gini"` | 分裂准则；也可以选择 `"entropy"` |
| `max_depth` | `None` | 树的最大深度；`None` 表示不由该参数限制深度 |
| `min_samples_split` | `2` | 一个内部节点继续分裂所需的最少样本数 |
| `min_samples_leaf` | `1` | 叶节点所需的最少样本数 |
| `random_state` | `None` | 控制随机过程 |
| `max_features` | `None` | 每次分裂时考虑的候选特征数或比例 |
| `ccp_alpha` | `0.0`、`0.01`、`0.1` | 成本复杂度剪枝参数，用于控制树的复杂度 |

#### 3.5 决策树结构图

```python
from sklearn.tree import plot_tree

plot_tree(tree_model, feature_names=X_train.columns,
          class_names=['No Diabetes', 'Diabetes'], filled=True)
```

![决策树结构示例](outputs/figures/models/tree_structure.jpg)

<a id="random-forest"></a>
### 4. 随机森林模型（Random Forest，RF）

#### 4.1 基本原理

随机森林基于决策树，采用随机方式建立一个由多棵决策树组成的森林。当新样本进入时，每棵决策树分别进行预测，再汇总得到分类结果。

分类森林通常采用多数投票汇总预测。

**建立森林的具体步骤：**

1. 从原始训练数据集中通过自助采样（Bootstrap Sampling），即有放回随机抽样，抽取一定数量的数据，组成子数据集。
2. 为每个子数据集训练一棵决策树。在决策树的每个决策节点，不使用所有特征，而是随机选取部分特征，并在这些特征中选择分裂规则。
3. 重复上述过程，得到多棵不同的决策树，对新样本汇总所有树的预测结果。

**为什么已经有决策树，还需要随机森林？**

核心原因是决策树算法存在容易过拟合、模型结构不稳定的缺点。随机森林通过袋装法（Bagging）集成多棵决策树，提高模型的泛化能力和稳健性。

单棵树的判断可能对某些样本变化非常敏感，多棵不同的树可以缓解这种不稳定。

**为什么使用随机抽样和随机特征选择？**

可以借助 Hoeffding 不等式理解：在个体分类错误相互独立等理想条件下，增加个体分类器数量可以降低集成分类错误。

实际中，个体分类器为解决同一个问题而训练，分类错误并不是完全独立的。虽然无法保证个体学习器之间完全独立，但可以设法让它们尽可能不同。随机抽样和随机特征选择就是这样的两种办法。

**核心超参数**

随机森林有两个重要超参数：决策树的数量，以及每次分裂随机选取的候选变量数。此外，树的最大深度、节点分裂最小样本数等也很重要。

#### 4.2 优点

- **预测表现相对稳定：** 通过集成多棵树的结果，减少对单棵树的依赖。
- **抗过拟合能力增强：** 随机抽样和特征子集选择产生多棵不同的树，降低单棵决策树容易过拟合的风险。
- **稳健性较好：** 其对部分异常值和噪声的影响相对不敏感。
- **能够并行训练：** 不同树之间的训练过程相对独立，可以并行训练多棵树。
- **袋外数据可用于结果评价：** Bootstrap 抽样后，未被某棵树抽中的样本可用于该树的袋外评价。未被抽中的样本比例约为 36.8%。

#### 4.3 缺点

- **计算开销大：** 需要训练大量决策树，数据量大时训练较慢。
- **模型解释性较差：** 相较单棵树，整个森林的决策过程不容易直接展示。
- **特征重要性可能存在偏好：** 某些特征重要性指标可能偏好取值较多的特征。

#### 4.4 Python 函数与常用参数

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=123, oob_score=True)
rf_model.fit(X_train, y_train)  # 训练随机森林
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]  # 预测类别 1 的概率
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `n_estimators` | `100` | 森林中决策树的数量 |
| `max_features` | `"sqrt"` | 每次节点分裂时考虑的候选特征数量；`"sqrt"` 表示特征总数的平方根 |
| `max_depth` | `None` | 单棵树的最大深度 |
| `oob_score` | `False` | 是否使用袋外样本评价模型；示例设置为 `True` |
| `random_state` | `None` | 控制随机抽样及建树过程 |
| `criterion` | `"gini"` | 节点分裂准则 |
| `n_jobs` | `-1` | 控制并行计算所使用的任务数 |

<a id="xgboost"></a>
### 5. XGBoost 模型（eXtreme Gradient Boosting）

#### 5.1 基本原理

XGBoost 是一种基于决策树的梯度提升集成学习算法。

核心思想是将多个弱分类器（决策树）组合成一个强分类器。后续决策树逐步改进前面模型的预测，并通过迭代优化损失函数减少误差。

| 集成思路 | 主要特点 |
| --- | --- |
| Bagging | 训练多组不同的基学习器，再汇总结果；随机森林属于这一思路 |
| Boosting | 按顺序加入新的基学习器，逐步改进前面组合模型的表现 |

#### 5.2 优点

- **预测表现通常较好：** 通过集成多棵决策树逐步改进模型。
- **具有控制过拟合的机制：** 可通过正则化项、树深度、叶节点数等限制模型复杂度。
- **对预处理的要求相对较少：** 这类树模型通常可以处理缺失值，也不需要为变量尺度进行归一化。

#### 5.3 缺点

- **计算开销较大：** 需要迭代训练多棵树；Boosting 的轮次之间存在顺序依赖。
- **对噪声和异常值可能敏感：** 较多噪声和异常值可能影响后续迭代。
- **模型解释性较差：** 多棵树组成的整体模型不如单棵树直观。
- **超参数调优较复杂：** 学习率、正则化系数、迭代次数、树深度等参数较多，且对结果有影响。

#### 5.4 Python 函数与常用参数

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=123, eval_metric='logloss')
xgb_model.fit(X_train, y_train)  # 训练 XGBoost
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # 预测类别 1 的概率
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `n_estimators` | `100` | 弱学习器的数量，需与其他复杂度参数共同考虑 |
| `learning_rate` | `0.1` | 学习率，控制每棵树对整体模型的贡献 |
| `max_depth` | `3` | 单棵树的最大深度 |
| `subsample` | `1` | 每轮训练使用的样本比例 |
| `colsample_bytree` | `1` | 每棵树使用的特征比例 |
| `eval_metric` | `"logloss"` | 评价指标；二分类可选择 `"logloss"` 或 `"auc"` |
| `use_label_encoder` | `False` | 旧版标签编码相关参数，当前示例不再使用 |
| `random_state` | `0` | 随机种子；示例代码设为 `123` |

<a id="lightgbm"></a>
### 6. LightGBM 模型（Light Gradient Boosting Machine）

#### 6.1 基本原理

LightGBM 和 XGBoost 同属于梯度提升树框架。LightGBM 通过引入互斥特征捆绑、直方分桶、叶子优先生长和单边梯度抽样等技术，降低训练的时间和内存开销，尤其适合较大的数据集。

**主要技术**

**直方分桶（Histogram-based Algorithm）**

把连续变量划分成多个桶，再在这些桶之间寻找分裂点。例如，将年龄划分为若干区间，可以减少候选分裂点的数量，从而降低计算量。

**互斥特征捆绑（Exclusive Feature Bundling，EFB）**

将一些不同特征中相互排斥的取值组合到较少的特征中。这一方法对高度稀疏的数据特别有用，可以减少特征数量和计算量。

**叶子优先生长（Leaf-wise Growth）**

传统的逐层生长方式会逐层寻找分裂点。叶子优先生长则优先分裂当前增益最大的叶节点，以减少不必要的分裂计算。

**单边梯度抽样（Gradient-based One-Side Sampling，GOSS）**

对样本进行不均匀抽样，保留梯度较大的样本，并按比例抽取梯度较小的样本，减少计算量。

#### 6.2 优点

- LightGBM 和 XGBoost 同属于 Boosting 算法，都能够学习复杂的非线性关系。LightGBM 的突出特点是训练效率和内存使用方面的优化。

#### 6.3 缺点

- LightGBM 仍需合理控制模型复杂度和超参数，也可能发生过拟合。

#### 6.4 Python 函数与常用参数

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(random_state=123)
lgb_model.fit(X_train, y_train)  # 训练 LightGBM
y_pred_prob = lgb_model.predict_proba(X_test)[:, 1]  # 预测类别 1 的概率
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `boosting_type` | `"gbdt"` | Boosting 类型，例如 `"gbdt"`、`"dart"`、`"goss"` |
| `num_leaves` | `31` | 单棵树的最大叶节点数，影响模型复杂度 |
| `max_depth` | `-1` | 最大树深度；`-1` 表示不由该参数限制 |
| `learning_rate` | `0.1` | 学习率，控制每轮树对模型的贡献 |
| `n_estimators` | `100` | 弱学习器的数量 |
| `subsample` | `1.0` | 每次抽样使用的样本比例 |
| `colsample_bytree` | `1.0` | 每棵树使用的特征比例 |
| `random_state` | `None` | 随机种子 |
| `subsample_freq` | 示例使用 `1` | 控制样本抽样频率；与 `subsample` 配合使用 |

<a id="svm"></a>
### 7. 支持向量机模型（Support Vector Machine，SVM）

#### 7.1 基本原理

支持向量机通过找到一个最优超平面将不同类别的数据分开。这个最优超平面尽可能增大不同类别之间的间隔。

对于二维数据，超平面是一条直线；对于三维数据，是一个平面；对于更高维数据，则是超平面。

离超平面近、对决策边界起关键作用的样本称为支持向量。最优超平面由这些支持向量决定。

#### 7.2 优点

- **较强的对高维数据的适应能力：** 可通过核技巧将数据映射到更高维空间，处理原始空间中难以线性分开的情况。
- **在部分小样本任务中表现较好：** 决策边界主要由支持向量决定。
- **较好的泛化潜力：** 最大化间隔并配合正则化，有助于控制过拟合。

#### 7.3 缺点

- **计算复杂度较高：** 对大样本数据，特别是核方法，训练可能较慢。
- **需要设置合适的超参数：** 惩罚系数、核函数类型及核参数对结果有影响。
- **解释性差：** 特别是使用核技巧后，难以直观解释分类结果。
- **内存开销可能较大：** 训练需要存储数据及核相关信息。

#### 7.4 Python 函数与常用参数

```python
from sklearn.svm import SVC

svm_model = SVC(probability=True, random_state=123)
svm_model.fit(X_train, y_train)  # 训练支持向量分类器
y_pred_prob = svm_model.predict_proba(X_test)[:, 1]  # 预测类别 1 的概率
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `C` | `1.0` | 正则化强度的倒数；较大时更强调训练样本的分类拟合 |
| `kernel` | `"rbf"` | 核函数类型，例如 `"linear"`、`"poly"`、`"rbf"`、`"sigmoid"` |
| `gamma` | `"scale"` | 核函数系数，影响相应核函数的作用范围 |
| `degree` | `3` | 多项式核的阶数，仅对 `kernel="poly"` 生效 |
| `probability` | `False` | 是否启用概率估计；设为 `True` 后可调用 `predict_proba()` |
| `random_state` | `None` | 在启用概率估计时控制相应随机过程 |

<a id="ann"></a>
### 8. 人工神经网络模型（Artificial Neural Network，ANN）

#### 8.1 基本原理

神经网络模拟人脑神经系统进行模式识别，由大量简单的处理单元——神经元——按照某种方式连接而成。

通过调整各层之间的连接权重，神经网络可以学习复杂的函数映射，从而从输入数据中提取特征并输出预测。

| 层次 | 作用 |
| --- | --- |
| 输入层 | 接收外部输入数据 |
| 隐藏层 | 进行特征提取和转换，可包含多层 |
| 输出层 | 输出模型结果 |

#### 8.2 优点

- **强大的建模能力：** 能处理高度复杂的数据，应用范围较广。
- **结构灵活：** 可以通过调整网络结构和参数适应不同的数据任务。
- **较强的非线性表达能力：** 通过多层变换学习复杂关系。

#### 8.3 缺点

- **计算资源需求较大：** 网络规模较大时，训练通常需要较多资源。
- **训练困难：** 模型复杂且样本不足时容易过拟合。
- **数据需求可能较高：** 特别是较大的深度学习模型，数据量会显著影响效果。
- **超参数调优困难：** 学习率、隐藏层数、神经元数、训练轮数等会影响性能。

#### 8.4 Python 函数与常用参数

```python
from sklearn.neural_network import MLPClassifier

ann_model = MLPClassifier(random_state=123, max_iter=500)
ann_model.fit(X_train, y_train)  # 训练多层感知机分类器
y_pred_prob = ann_model.predict_proba(X_test)[:, 1]  # 预测类别 1 的概率
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `hidden_layer_sizes` | `(100,)` | 每个隐藏层的神经元数量；例如 `(50, 30)` 表示两个隐藏层 |
| `activation` | `"relu"` | 激活函数，可选 `"relu"`、`"tanh"`、`"logistic"` 等 |
| `solver` | `"adam"` | 权重优化方法，例如 `"adam"`、`"sgd"`、`"lbfgs"` |
| `alpha` | `0.0001` | L2 正则化参数，用于控制模型复杂度 |
| `learning_rate` | `"constant"` | 学习率调整方式，例如 `"constant"`、`"invscaling"`、`"adaptive"` |
| `max_iter` | `200` | 最大迭代次数；示例设置为 `500` |
| `random_state` | `None` | 控制随机初始化等过程 |

---

<a id="evaluation"></a>
## 三、模型预测效果评价

<a id="evaluation-overview"></a>
### 1. 预测效果评价方式

模型评价既包括数值指标，也包括评价图形。

| 类别 | 内容 |
| --- | --- |
| 分类指标 | 准确率、精确率、灵敏度／召回率、特异度、F1 分数 |
| 区分能力 | ROC 曲线与 AUC |
| 概率可靠性 | 校准曲线 |
| 阈值下的决策表现 | 决策曲线分析（DCA） |
| 分类结果汇总 | 混淆矩阵 |

这些指标和图形回答的问题不同，不应只依靠一个数值判断模型是否“好”。

<a id="confusion-matrix"></a>
### 2. 混淆矩阵及分类指标

#### 2.1 混淆矩阵（Confusion Matrix）

混淆矩阵用于总结分类模型的预测结果。

| 名称 | 含义 |
| --- | --- |
| 真阳性（True Positive，TP） | 实际为正类，被预测为正类的样本数 |
| 真阴性（True Negative，TN） | 实际为负类，被预测为负类的样本数 |
| 假阳性（False Positive，FP） | 实际为负类，被预测为正类的样本数 |
| 假阴性（False Negative，FN） | 实际为正类，被预测为负类的样本数 |

按类别顺序 `[0, 1]`，混淆矩阵排列如下：

| 实际类别 \ 预测类别 | 阴性（0） | 阳性（1） |
| --- | --- | --- |
| 阴性（0） | TN | FP |
| 阳性（1） | FN | TP |

因此：

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 行为实际类别，列为预测类别
tn, fp, fn, tp = cm.ravel()  # 顺序为 TN、FP、FN、TP
```

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `y_true` | 必须提供 | 实际标签 |
| `y_pred` | 必须提供 | 预测类别，不是预测概率 |
| `labels` | `None` | 指定矩阵中的类别顺序；二分类可明确设为 `[0, 1]` |
| `sample_weight` | `None` | 样本权重 |
| `normalize` | `None` | 是否对计数归一化；可按真实类别、预测类别或全部样本进行归一化 |

#### 2.2 准确率（Accuracy）

准确率是分类模型正确分类的样本数与样本总数之比，反映模型整体的分类情况。

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+FN+TN+FP}
$$

它回答的是：**所有样本中，有多少被模型正确分类？**

#### 2.3 精确率（Precision）

精确率是在所有预测为正类的样本中，实际为正类的比例。

$$
\mathrm{Precision}=\frac{TP}{TP+FP}
$$

它回答的是：**模型判定为阳性的样本中，有多少是真正的阳性？**

#### 2.4 灵敏度／召回率（Sensitivity／Recall）

灵敏度是在所有实际为正类的样本中，模型正确预测为正类的比例，反映模型捕捉正类样本的能力。

$$
\mathrm{Sensitivity}=\mathrm{Recall}=\frac{TP}{TP+FN}
$$

它回答的是：**真正的阳性样本中，有多少被模型找到了？**

#### 2.5 F1 分数（F1 Score）

F1 分数是精确率和召回率的调和平均，综合考虑模型的精确率和召回率。

$$
\mathrm{F1}=2\times\frac{\mathrm{Precision}\times\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
=\frac{2TP}{2TP+FP+FN}
$$

它回答的是：**综合考虑精确率和召回率时，模型的表现如何？**

#### 2.6 特异度（Specificity）

特异度是在所有实际为负类的样本中，模型正确预测为负类的比例，反映模型对负类样本的识别能力。

$$
\mathrm{Specificity}=\frac{TN}{TN+FP}
$$

它回答的是：**真正的阴性样本中，有多少被模型正确识别为阴性？**

#### 2.7 基于混淆矩阵计算评价指标

**Python 计算函数**

```python
def calculate_acc_pre_sen_f1_spc(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    accuracy = (tp + tn) / (tp + fn + tn + fp)  # 准确率
    precision = tp / (tp + fp)  # 精确率
    sensitivity = tp / (tp + fn)  # 灵敏度
    f1score = 2 * (precision * sensitivity) / (precision + sensitivity)  # F1 分数
    specificity = tn / (tn + fp)  # 特异度
    return accuracy, precision, sensitivity, f1score, specificity
```

| 输入／输出 | 说明 |
| --- | --- |
| 输入 | 形状为 $`2\times2`$ 的混淆矩阵，排列为 `[[TN, FP], [FN, TP]]` |
| 输出 | 准确率、精确率、灵敏度、F1 分数、特异度，按该顺序返回 |

#### 2.8 混淆矩阵读图示例

![随机森林测试集混淆矩阵](outputs/figures/evaluation/confusion_matrix_rf_test.png)

图中 $`TN=117`$、$`FP=9`$、$`FN=30`$、$`TP=33`$，因此：

$$
\mathrm{Accuracy}=\frac{117+33}{189}\approx0.794
$$

$$
\mathrm{Sensitivity}=\frac{33}{33+30}\approx0.524,\qquad
\mathrm{Specificity}=\frac{117}{117+9}\approx0.929
$$

这个例子表明，模型可以同时具有较高的特异度和相对较低的灵敏度。混淆矩阵反映的是**一个具体分类阈值下**的结果。

<a id="roc-auc"></a>
### 3. ROC 曲线与 AUC

#### 3.1 ROC 曲线（Receiver Operating Characteristic Curve）

ROC 曲线描绘真正率（True Positive Rate，TPR）与假正率（False Positive Rate，FPR）的关系，适用于评价二分类模型的区分能力。

$$
\mathrm{TPR}=\frac{TP}{TP+FN}
$$

$$
\mathrm{FPR}=\frac{FP}{FP+TN}=1-\mathrm{Specificity}
$$

通过不同的分类阈值可以绘制出 ROC 曲线，用于比较不同模型的区分能力。

#### 3.2 AUC（Area Under the Curve）

AUC 是 ROC 曲线下面积，取值在 0–1 之间。在正类方向一致的前提下，越接近 1，表示模型的区分能力越强。

AUC 主要评价模型的**区分能力**。

#### 3.3 优点与局限

**优点**

- 直观，易于比较不同模型的区分能力。
- 能展示模型在不同阳性阈值下的表现。
- 相较单一准确率，在类别不平衡时仍有评价价值。

**局限**

- 不直接考虑假阳性与假阴性的不同代价。
- 对极不平衡的数据，可能存在 AUC 较高、实际使用效果仍不理想的情况。

#### 3.4 Python 函数与常用参数

```python
from sklearn.metrics import roc_curve, roc_auc_score

auc_value = roc_auc_score(y_true, y_pred_prob)  # 计算 AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)  # 计算各阈值下的 ROC 坐标
```

| 函数／参数 | 说明 |
| --- | --- |
| `roc_auc_score(y_true, y_score)` | 根据实际标签和预测分数计算 AUC；本例传入正类概率 |
| `y_true` | 实际标签，二分类示例中为 0／1 |
| `y_score` | 正类预测概率或符合接口要求的连续决策分数，不能替换为只有 0／1 的分类结果来表达同样的信息 |
| `average` | 多类别／多标签任务中涉及的平均方式；普通二分类不需要为此额外设置 |
| `multi_class` | 多分类任务的处理方式；本二分类示例不使用 |
| `max_fpr` | 可用于指定部分 AUC 的范围；完整 AUC 示例不设置 |
| `roc_curve(y_true, y_score)` | 返回 FPR、TPR 和对应的阈值 |

**ROC 绘图函数**

```python
def ROC_plot(y_label, y_pred_prob, auc_value):
    fpr, tpr, _ = roc_curve(y_label, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')  # 随机区分参考线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
```

#### 3.5 AUC 的 Bootstrap 置信区间

Bootstrap 通过对评价数据重复有放回抽样，观察评价指标在重抽样中的变化。

当前代码使用 **2,000 次分层 Bootstrap 百分位区间**：

1. 分别在阳性样本和阴性样本内部有放回抽样，保留两类样本数。
2. 将被抽中的真实标签与对应预测概率一起取出，重新计算 AUC。
3. 重复 2,000 次，得到 AUC 的重抽样分布。
4. 取第 2.5% 和第 97.5% 分位数，作为 95% 区间的上下限。

```math
\mathrm{CI}_{95\%}=\left[Q_{0.025}(\mathrm{AUC}^{*}),\ Q_{0.975}(\mathrm{AUC}^{*})\right]
```

这里重抽样的是已经得到的预测结果，不重新训练模型。它描述固定模型在评价样本抽样下的不确定性，不包含完整训练流程重新拟合带来的全部变化。

在一人一行的数据中，可按行进行这一操作；有重复测量或同一患者多行记录时，不能把这些行当作相互独立的患者。

验证集已经用于调参，其区间不能消除模型选择带来的乐观性；独立测试集的评价角色与验证集不同。

#### 3.6 ROC 读图示例

![七种模型的测试集 ROC 曲线](outputs/figures/evaluation/ROC_curves_allmodel_test.png)

读图时，横轴为假正率（FPR），纵轴为真正率（TPR）。在相同假正率下，曲线越高，说明模型能够识别更多阳性样本；曲线整体越靠近左上角，区分能力通常越强。灰色对角虚线是随机分类的参考线。

本图中，SVM 的测试集 AUC 为 0.891，决策树为 0.763。可以结合图例中的 AUC 和曲线位置，比较七种模型的区分能力。

<a id="calibration"></a>
### 4. 校准曲线（Calibration Curve）

#### 4.1 概念与作用

校准曲线将模型预测的概率与实际观察到的结果进行比较，用于评估模型预测概率的准确性。

它的作用是帮助判断模型输出的概率是否可靠。

如果曲线接近 45 度对角线，说明相应概率范围中的预测与观察较为接近；如果明显偏离对角线，说明模型的概率预测可能存在偏差。

校准曲线在需要概率输出的应用中尤其重要。

#### 4.2 图形特点

| 图形元素 | 含义 |
| --- | --- |
| 横轴 | 每个分箱内的平均预测概率 |
| 纵轴 | 同一分箱内的实际阳性比例 |
| 45 度参考线 | 理想校准状态，即平均预测概率与观察比例一致 |
| 曲线上的点 | 各个非空概率分箱的汇总结果 |

#### 4.3 优点与局限

**优点**

- 评估模型预测概率的准确性。
- 帮助发现概率输出偏差，尤其适合重视概率可靠性的应用场景。

**局限**

- 适用于概率预测结果，不适合只有硬分类标签而没有概率输出的情形。
- 当数据较为复杂或噪声较大时，图形解释可能变得困难。
- 小样本下各个分箱中的观测数量可能有限，曲线会随分箱方式产生变化。

#### 4.4 Python 函数与常用参数

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_true, y_pred_prob, n_bins=10, strategy='uniform'
)
plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
```

| 参数／返回值 | 示例值／含义 | 说明 |
| --- | --- | --- |
| `y_true` | 必须提供 | 实际标签 |
| `y_prob` | 必须提供 | 预测为正类的概率 |
| `n_bins` | 示例使用 `10` | 将概率分成多少个区间 |
| `strategy` | `"uniform"` | 等宽分箱；`"quantile"` 表示按分位数分箱 |
| `normalize` | 旧版参数 | 当前示例不使用 |
| `prob_true` | 返回值 | 各非空分箱中的实际阳性比例 |
| `prob_pred` | 返回值 | 各非空分箱中的平均预测概率 |

`CaliC_plot(y_label, y_pred_prob, n_bins=10)` 是对校准曲线计算与绘图过程的封装。

#### 4.5 校准曲线读图示例

![七种模型的测试集校准曲线](outputs/figures/evaluation/Calibration_curves_allmodel_test.png)

读图时，横轴是模型给出的平均预测概率，纵轴是相应分箱内实际观察到的阳性比例。曲线上的点越接近灰色对角线，表示该分箱的预测概率越接近实际比例；点位于对角线上方表示模型低估了阳性发生比例，下方则表示高估。

例如，某个分箱的平均预测概率为 0.6，实际阳性比例为 0.8，该点就位于对角线上方。本图中不同模型的曲线均有波动，读图时重点关注它们与对角线的偏离程度。

<a id="dca"></a>
### 5. 决策曲线分析（Decision Curve Analysis，DCA）

#### 5.1 概念与作用

决策曲线分析是一种评估分类模型在实际应用中的临床价值和实用性的统计方法。

与准确率、AUC 等传统评价指标不同，DCA 考虑决策的临床后果，特别是在不同阈值下的预期收益，以衡量模型在实际应用中的效用。

它将模型预测与决策过程的后果相结合，通过计算不同阈值下的净收益（Net Benefit）来评价模型的实用价值。

#### 5.2 阈值概率与净收益

**阈值概率（Threshold Probability）** 是采取某项行动的概率界限。例如，以 0.2 为阈值，表示预测概率超过该值时按阳性策略处理。

净收益计算公式：

$$
\mathrm{NB}(p_t)=\frac{TP(p_t)}{N}
-\frac{FP(p_t)}{N}\times\frac{p_t}{1-p_t}
$$

其中：

| 符号 | 含义 |
| --- | --- |
| $`p_t`$ | 阈值概率 |
| $`N`$ | 评价样本总数 |
| $`TP(p_t)`$ | 在该阈值下的真阳性数量 |
| $`FP(p_t)`$ | 在该阈值下的假阳性数量 |
| $`p_t/(1-p_t)`$ | 该阈值对应的假阳性代价权重 |

**净收益不是准确率，也不是直接的经济利润。** 它是在所设定阈值与收益／代价权衡下的评价量。

#### 5.3 图形中的三类曲线

| 曲线 | 含义 |
| --- | --- |
| 模型曲线 | 基于模型预测，在不同阈值下采取决策时的净收益 |
| Treat All | 假定所有人都按阳性策略处理 |
| Treat None | 假定所有人都按阴性策略处理，净收益为 0 |

设样本阳性比例为 $`\pi`$，则：

$$
\mathrm{NB}_{\mathrm{all}}(p_t)=\pi-(1-\pi)\frac{p_t}{1-p_t}
$$

$$
\mathrm{NB}_{\mathrm{none}}(p_t)=0
$$

#### 5.4 优点与局限

**优点**

- **考虑决策后果：** 不仅依靠 AUC、准确率等传统指标，也考虑分类错误的代价。
- **阈值灵活：** 可在不同阈值下评价净收益。
- **支持模型比较：** 可在同一阈值范围内比较多个模型。

**局限**

- **需要合理的阈值范围：** 在实际应用场景中，合理阈值可能不明确。
- **依赖数据和模型质量：** 数据质量、预测概率可靠性等会影响结果。
- **需结合具体应用解释：** 在模拟数据上计算的净收益，只能用于方法演示，不能证明真实临床获益。

#### 5.5 Python 计算与函数参数

可使用自定义函数 `calculate_net_benefi()`，分别返回模型、Treat All 和 Treat None 的净收益序列。

```python
thresholds = np.linspace(0.01, 0.99, 100)  # 设置阈值范围
net_benefit_model, net_benefit_alltrt, net_benefits_notrt = calculate_net_benefi(
    y_true, y_pred_prob, thresholds=thresholds
)
```

| 参数／返回值 | 说明 |
| --- | --- |
| `y_label` | 实际标签，二分类示例中为 0／1 |
| `y_pred_prob` | 预测正类概率 |
| `thresholds` | 需要评价的阈值序列 |
| `net_benefit_model` | 基于模型的净收益序列 |
| `net_benefit_alltrt` | Treat All 的净收益序列 |
| `net_benefits_notrt` | Treat None 的净收益序列，全部为 0 |
| `DCA_plot(...)` | 将三组净收益与相同的阈值序列绘制为图形 |

#### 5.6 DCA 读图示例

![七种模型的测试集决策曲线](outputs/figures/evaluation/DCA_curves_allmodel_test.png)

读图时，先在横轴选择一个阈值概率，例如 0.20，再沿竖直方向比较该阈值下各条曲线的净收益。纵轴越高，表示净收益越大；如果模型曲线同时高于红色虚线（Treat All）和绿色虚线（Treat None），说明在该阈值下，按模型预测采取行动优于全部按阳性处理或全部按阴性处理。

本图中，阈值约为 0.20 时，多数模型的净收益高于两条参考线。红色 Treat All 曲线在约 0.33 处与零线相交；随着阈值升高，模型净收益整体趋于降低。图中红色虚线在较高阈值处消失，是因为它的净收益低于纵轴显示下限（−0.2），而不是计算中断。

<a id="evaluation-summary"></a>
### 6. 预测效果评价总结

| 指标／图形 | 主要回答的问题 | 优点 | 主要局限 |
| --- | --- | --- | --- |
| 混淆矩阵 | 在某个阈值下，哪些样本预测正确、哪些预测错误？ | 直接展示 TP、TN、FP、FN | 依赖分类阈值 |
| Accuracy、Precision、Sensitivity、Specificity、F1 | 整体准确性及阳性／阴性的识别情况如何？ | 含义明确，便于计算 | 不同指标强调不同方面，不能彼此替代 |
| ROC／AUC | 模型区分阳性与阴性的能力如何？ | 可跨阈值观察区分能力 | 不直接评价概率校准，也不直接纳入决策代价 |
| 校准曲线 | 预测概率与实际发生比例是否匹配？ | 适合检查概率可靠性 | 受样本数量、分箱及噪声影响 |
| DCA | 在不同阈值与收益／代价权衡下，模型是否有净收益？ | 将预测与决策后果相联系 | 需要合理阈值与明确应用场景 |

**区分能力、概率校准、分类表现与决策收益，是相关但不同的评价维度。**

---

<a id="interpretability"></a>
## 四、机器学习可解释性

<a id="interpretability-purpose"></a>
### 1. 机器学习可解释性的由来

许多常见的机器学习模型，例如深度神经网络、随机森林、XGBoost 等，常被视为“黑箱”模型，即难以用简单、直观的方式解释其预测决策过程。

缺乏透明度和可解释性的现象，限制了机器学习在一些敏感领域中的应用。

机器学习模型可解释性研究，对提高模型的透明度和可信度、帮助优化模型、发现潜在偏见及更好地应用模型具有重要意义。

<a id="interpretability-methods"></a>
### 2. 机器学习可解释性方法

| 方法 | 介绍 |
| --- | --- |
| SHAP（SHapley Additive exPlanations） | 基于博弈论中的 Shapley 值，为每个特征计算其对模型输出的贡献；提供全局和局部解释 |
| LIME（Local Interpretable Model-agnostic Explanations） | 在模型预测附近训练一个简单模型，解释原模型的局部决策；侧重局部解释 |
| 模型透明化与可视化技术 | 决策树等模型本身具有一定直观可解释性；特征重要性、部分依赖图等方法可帮助理解其他模型 |

下面重点介绍 SHAP。

<a id="shap-values"></a>
### 3. SHAP 值及其计算

#### 3.1 SHAP 值的含义

SHAP 的目标是为每个特征分配一个贡献值，用来表示该特征在特定预测中的作用。

SHAP 提供两种层次的解释：

| 层次 | 含义 |
| --- | --- |
| 全局解释 | 汇总一组样本的特征贡献，例如特征重要性，了解特征对模型预测的总体作用 |
| 局部解释 | 为某个单独个体的预测提供解释，了解不同特征在该次预测中的作用 |

SHAP 可适用于多种类型的模型，包括线性模型和机器学习模型。以下介绍模型无关的 KernelExplainer。

#### 3.2 基线与特征贡献

瀑布图和力图表达的是：从一个参考预测出发，各特征的贡献使结果向最终预测移动。

对于当前解释概率的示例，可写为：

$$
f(x)=E_{\mathrm{background}}[f(X)]+\sum_{j=1}^{p}\phi_j(x)
$$

其中，$`E_{\mathrm{background}}[f(X)]`$ 为背景样本上的参考预测，$`\phi_j(x)`$ 为第 $`j`$ 个特征对该次预测的 SHAP 贡献。

正 SHAP 值使解释目标高于参考值，负 SHAP 值使解释目标低于参考值。

#### 3.3 KernelExplainer

KernelExplainer 是模型无关的 SHAP 解释器，可以通过模型的预测函数计算特征归因。它基于局部线性近似进行解释。

建立解释器需要两个核心输入：**模型预测函数**和**背景数据**。

```python
import shap

np.random.seed(123)  # 固定近似计算的随机种子
background_data = shap.sample(X_train, 50, random_state=123)  # 从训练数据中取背景样本
explainer = shap.KernelExplainer(
    lambda X: model.predict_proba(pd.DataFrame(X, columns=X_train.columns)),
    background_data
)
shap_values = explainer(X_test.iloc[:50, :])  # 解释选定的测试样本
```

| 参数 | 示例值／用法 | 说明 |
| --- | --- | --- |
| `model` | 必须提供 | 要解释的模型函数，例如返回类别概率的函数 |
| `data` | 必须提供 | 背景数据，用于确定参考分布；可使用训练数据的一部分 |
| `link` | `"identity"` | 解释值与输出之间的链接；也可使用 `"logit"`，其解释尺度为对数优势 |
| `feature_names` | `None` | 特征名称；使用 DataFrame 时可从列名取得 |
| `seed` | `None` | 随机性相关参数；示例采用 `np.random.seed()` 并固定背景抽样种子 |

| 计算时的参数 | 示例值／用法 | 说明 |
| --- | --- | --- |
| `X` | 必须提供 | 需要解释的一个或多个样本 |
| `l1_reg` | `"num_features(10)"` | 近似估计时的正则化／特征选择设置；不是绘图显示数量 |
| `silent` | `False` | 是否隐藏计算过程的进度信息 |

#### 3.4 返回对象与维度

调用解释器后得到 `shap.Explanation` 对象，其中包含 SHAP 值、基线值、样本特征值和特征名等信息。

对当前返回两个类别概率的分类模型：

| 写法 | 含义 |
| --- | --- |
| `shap_values[:, :, 1]` | 所有被解释样本、所有特征对类别 1 概率的贡献 |
| `shap_values[:, "age", 1]` | 所有被解释样本中 `age` 对类别 1 概率的贡献 |
| `shap_values[obs_index, :, 1]` | 某一条观测的全部特征贡献 |
| `shap_values[:, :, 1].values` | 相应的数值数组 |
| `shap_values[:, :, 1].base_values` | 相应的参考输出 |
| `shap_values[:, :, 1].data` | 对应样本的特征取值 |

Logistic 的预测函数直接返回类别 1 的一维概率，其 SHAP 对象为“样本 × 特征”，因此不需要再取 `[:, :, 1]`。

下面各类图统一以返回双类别概率的随机森林为例。代码中的 `obs_index=5` 表示被解释样本中的第六条观测，不是原始数据索引为 5 的记录。

#### 3.5 背景样本与解释样本

背景样本用于提供比较的参考分布；解释样本是需要解释其预测结果的样本。两者用途不同。

本节配图以 50 个训练样本作为背景数据，解释测试集前 50 个样本。

<a id="shap-bar"></a>
### 4. SHAP 特征重要性条形图（Bar Plot）

#### 4.1 作用

提供全局特征重要性的简单展示，按特征重要性排序。

#### 4.2 图形特点

| 图形元素 | 含义 |
| --- | --- |
| 横轴 | SHAP 值绝对值的平均值，表示所解释样本中的平均贡献强度 |
| 纵轴 | 特征名称，按重要性降序排列 |
| 条形长度 | 特征重要性的大小 |

对特征 $`j`$，其平均绝对 SHAP 值可写为：

$$
I_j=\frac{1}{n}\sum_{i=1}^{n}|\phi_{ij}|
$$

#### 4.3 图形解读

读图时，先看纵轴的特征排序，再看横轴 `mean(|SHAP value|)`。条形越长，表示该特征在这组样本中对模型输出的**平均贡献幅度**越大。

本图中，`glucose` 排名第一，平均绝对 SHAP 值约为 0.12；`hyperlip` 和 `age` 次之，均约为 0.05。这说明随机森林在解释这 50 个测试样本时，`glucose` 对预测概率的平均影响幅度最大。

**条形图只比较贡献大小，不显示正负方向。** 即使条形向右延伸，也不能据此判断该特征会提高类别 1 的预测概率；需要结合蜂群图查看贡献方向。

#### 4.4 Python 函数与常用参数

```python
shap.plots.bar(shap_values[:, :, 1], max_display=15, show=False)
plt.savefig('outputs/figures/shap/bar_rf.png', dpi=200, bbox_inches='tight')
plt.close()
```

| 参数 | 示例值／用法 | 说明 |
| --- | --- | --- |
| `shap_values` | 必须提供 | 一组样本的 SHAP Explanation，二分类例中选取类别 1 |
| `max_display` | 示例为 `15` | 最多展示多少个特征 |
| `order` | `None` | 用于控制排序；示例沿用库的排序方式 |
| `show` | `True`；保存图片时设为 `False` | 是否立即调用显示，以便继续保存或调整图形 |

#### 4.5 示例图

![随机森林 SHAP 特征重要性条形图](outputs/figures/shap/bar_rf.png)

<a id="shap-beeswarm"></a>
### 5. SHAP 特征重要性蜂群图（Beeswarm Plot）

#### 5.1 作用

全局展示模型中特征的影响力，以及不同特征取值对应的贡献分布趋势。

#### 5.2 图形特点

| 图形元素 | 含义 |
| --- | --- |
| 每个点 | 某一个样本在该特征上的解释 |
| 横轴 | SHAP 值，表示相对于参考输出的正负贡献 |
| 纵轴 | 特征名称，通常按平均绝对 SHAP 值排序 |
| 点的颜色 | 特征取值的大小，而不是 SHAP 值本身的大小 |
| 横向分布范围 | 该特征贡献在不同样本中的变化 |

#### 5.3 图形解读

读图时，以横轴的 **SHAP = 0** 为界：点位于右侧（正 SHAP 值），表示该特征使类别 1 的预测概率升高；点位于左侧（负 SHAP 值），表示使预测概率降低。红色表示特征值较高，蓝色表示特征值较低。同一行的点越分散，说明该特征对不同样本的贡献差异越大。

本图中，`glucose` 的高值（红色）主要分布在右侧，低值（蓝色）主要分布在左侧，说明较高的 `glucose` 通常对应正向预测贡献。`exercise` 呈现相反的分布：较高取值的点主要位于左侧，较低取值的点主要位于右侧。

因此，**条形图用于判断哪些特征重要，蜂群图进一步说明特征取值与贡献方向之间的关系，以及不同样本的差异。**

#### 5.4 Python 函数与常用参数

```python
shap.plots.beeswarm(shap_values[:, :, 1], max_display=15, show=False)
plt.savefig('outputs/figures/shap/beeswarm_rf.png', dpi=200, bbox_inches='tight')
plt.close()
```

| 参数 | 示例值／用法 | 说明 |
| --- | --- | --- |
| `shap_values` | 必须提供 | 指定类别对应的一组样本解释 |
| `max_display` | `20`；示例为 `15` | 最多展示的特征数量 |
| `order` | 按重要性排序 | 控制特征的显示顺序 |
| `show` | `True`；保存时为 `False` | 是否立即显示图形 |

#### 5.5 示例图

![随机森林 SHAP 蜂群图](outputs/figures/shap/beeswarm_rf.png)

<a id="shap-dependence"></a>
### 6. SHAP 依赖性图（Dependence Plot）

#### 6.1 作用

展示某个特征的 SHAP 值与其自身取值之间的关系，并通过另一特征的颜色信息，帮助观察可能的交互模式。

#### 6.2 图形特点

| 图形元素 | 含义 |
| --- | --- |
| 横轴 | 该特征的取值 |
| 纵轴 | 该特征对应的 SHAP 值 |
| 每个点 | 一个样本 |
| 点的颜色 | 所指定的另一个特征的取值 |

#### 6.3 图形解读

读图时，横轴是 `age` 的特征取值，纵轴是 `age` 对类别 1 预测概率的 SHAP 贡献。纵轴高于 0 表示 `age` 推高预测概率，低于 0 表示降低预测概率。本图使用的是标准化后的 `age`，因此横轴的 0 对应训练集平均年龄，而不是 0 岁。

本图中，较低的 `age` 值通常对应负 SHAP 值；随着 `age` 增加，SHAP 值整体趋向正值，说明随机森林对年龄的预测贡献呈现总体上升的趋势。点并非完全落在一条曲线上，反映不同样本的贡献仍有差异。

点的颜色对应另一个特征 `gender` 的取值（蓝色较低、红色较高），可以在观察年龄与其 SHAP 贡献关系的同时，比较不同 `gender` 编码下散点的分布情况。

#### 6.4 Python 函数与常用参数

```python
shap.plots.scatter(
    shap_values[:, 'age', 1],
    color=shap_values[:, 'gender', 1],
    show=False
)
plt.savefig('outputs/figures/shap/scatter_rf.png', dpi=200, bbox_inches='tight')
plt.close()
```

| 参数 | 示例值／用法 | 说明 |
| --- | --- | --- |
| `shap_values` | 必须提供 | 通常传入一个特征对应的 Explanation 列 |
| `color` | 可指定另一特征 | 控制散点着色；传入 Explanation 列时，以该列的特征取值着色 |
| `axis_color` | `"#333333"` | 坐标轴颜色 |
| `cmap` | 使用库的色图 | 散点颜色映射 |
| `show` | `True`；保存时为 `False` | 是否立即显示图形 |

#### 6.5 示例图

![随机森林 SHAP 依赖性图](outputs/figures/shap/scatter_rf.png)

<a id="shap-waterfall"></a>
### 7. SHAP 瀑布图（Waterfall Plot）

#### 7.1 作用

解释单个样本的预测，展示不同特征从基线到最终预测结果的累积贡献。

#### 7.2 图形特点

- 显示从基线值到最终预测值的变化。
- 每一步变化由一个特征的 SHAP 值决定。
- 每个条形表示一个特征贡献，条形的长度反映贡献幅度。
- 特征标签可同时展示该样本的特征取值。

#### 7.3 图形解读

读图时，从下方的基线值 **`E[f(X)]`** 出发，将各特征的 SHAP 贡献逐项累加，最终到达上方的 **`f(x)`**，即该样本的模型预测概率。红色条形表示正向贡献，蓝色条形表示负向贡献；条形越长，贡献幅度越大。

本图中，基线概率约为 0.357，最终预测类别 1 的概率约为 0.076。`glucose`、`bmi`、`pregnant`、`insulin` 等特征的蓝色负向贡献较大；虽然 `pressure` 等特征具有一定正向贡献，但不足以抵消负向贡献，因此最终预测概率明显低于基线。

图中特征名称旁的数值（例如 `-0.188 = glucose`）是该样本的**特征取值**，而彩色条形上的 `-0.07` 等数字是对应的 **SHAP 贡献值**。

#### 7.4 Python 函数与常用参数

```python
obs_index = 5  # 选定第六条被解释观测
shap.plots.waterfall(shap_values[obs_index, :, 1], max_display=15, show=False)
plt.savefig(f'outputs/figures/shap/waterfall_{obs_index}_rf.png',
            dpi=200, bbox_inches='tight')
plt.close()
```

| 参数 | 示例值／用法 | 说明 |
| --- | --- | --- |
| `shap_values` | 单个样本的解释 | 必须是一条观测、一个模型输出对应的特征贡献 |
| `max_display` | `10`；示例为 `15` | 最多展示的特征数量 |
| `show` | `True`；保存时为 `False` | 是否立即显示图形 |

#### 7.5 示例图

![随机森林单个样本的 SHAP 瀑布图](outputs/figures/shap/waterfall_5_rf.png)

<a id="shap-force"></a>
### 8. SHAP 力图（Force Plot）

#### 8.1 作用

力图可以理解为一种“力学平衡系统”：各特征像不同方向的推力，把预测从参考值推向最终结果。

#### 8.2 图形特点

力图与瀑布图都展示单个样本的特征贡献，但排布方式不同。力图将正向、负向贡献在同一方向轴上组织起来，便于查看哪些因素将预测推高或拉低。也可以绘制多个预测的力图，每一列代表一个样本的 SHAP 分解。

#### 8.3 图形解读

**单个样本的力图：** 先找到 `base value`（基线预测）和 `f(x)`（该样本的预测结果）。红色表示将类别 1 的预测概率推高的贡献，蓝色表示将预测概率拉低的贡献。本图的基线约为 0.357，最终预测约为 0.08；`glucose`、`bmi`、`pregnant` 和 `insulin` 等蓝色负向贡献占主导，因此结果低于基线。颜色块中的特征名称和数值表示该样本的输入特征信息。

**多个样本的力图：** 横轴依次展示多个被解释样本，红色和蓝色区域分别反映推高、拉低预测的特征贡献。不同位置的起伏表明各样本的预测及其贡献构成存在差异。在 HTML 交互版本中可以查看具体样本及相应特征。

#### 8.4 Python 函数与常用参数

```python
force_plot_sig = shap.plots.force(shap_values[obs_index, :, 1])  # 单个样本的力图
shap.save_html(f'outputs/figures/shap/force_plot_sig_{obs_index}_rf.html',
               force_plot_sig)

force_plot_all = shap.plots.force(shap_values[:, :, 1])  # 所有被解释样本的力图
shap.save_html('outputs/figures/shap/force_plot_all_rf.html', force_plot_all)
```

| 参数／方法 | 示例值／用法 | 说明 |
| --- | --- | --- |
| `base_value` | 基线值 | 使用原始数组接口时提供参考输出；传入 Explanation 时可由对象取得 |
| `shap_values` | 一个或多个样本的贡献 | 指定对应类别或模型输出 |
| `features` | `None` | 特征取值，用于显示具体数值信息 |
| `feature_names` | `None` | 特征名列表，便于识别变量 |
| `matplotlib` | `False` | 是否使用 Matplotlib 绘制静态图；示例采用交互式图 |
| `shap.save_html()` | 文件路径和力图对象 | 将交互式力图保存为 HTML |

#### 8.5 示例图

**单个样本的力图**

![随机森林单个样本的 SHAP 力图](outputs/figures/shap/force_plot_sig_5_rf.png)

**50 个解释样本的力图**

![随机森林 50 个样本的 SHAP 力图](outputs/figures/shap/force_plot_all_rf.png)

交互式版本：[单个样本](outputs/figures/shap/force_plot_sig_5_rf.html) · [50 个样本](outputs/figures/shap/force_plot_all_rf.html)

<a id="shap-summary"></a>
### 9. SHAP 图形总结

| 图形 | 解释层次 | 主要用途 |
| --- | --- | --- |
| Bar Plot | 全局汇总 | 比较平均绝对贡献大小 |
| Beeswarm Plot | 全局分布 | 同时观察贡献幅度、方向、样本分布和特征取值 |
| Dependence Plot | 特征层面 | 查看特征取值与其贡献的关系，并观察潜在交互线索 |
| Waterfall Plot | 单个样本 | 展示从参考输出到最终预测的累积贡献 |
| Force Plot | 单个或多个样本 | 以相反方向的贡献展示预测变化，可交互查看 |
