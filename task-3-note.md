# task-3-note

### PS：这里通过看深度学习的数学这本书补充了偏导，链式法则等数学知识

如果有复合函数：

y=f(g(x))*y*=*f*(*g*(*x*))

设中间变量 u=g(x)*u*=*g*(*x*)，则 y=f(u)*y*=*f*(*u*) 

链式法则说：




$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$
多变量
$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\cdot\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\cdot\frac{\partial v}{\partial x}
$$

$$
\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u}\cdot\frac{\partial u}{\partial y} + \frac{\partial z}{\partial v}\cdot\frac{\partial v}{\partial y}
$$





应用梯度下降法时需要用到多变量函数的近似公式



#### 布骤

a = x * y        # 第一步：乘法
b = x * x        # 第二步：x的平方（等价于 x²）
c = a + b        # 第三步：加法
f = c * c        # 第四步：平方

#### 计算图

x ──┐
    ├──→ (×) ──→ a ──┐
y ──┘                │
                     ├──→ (+) ──→ c ──┐
x ──┐                │                 │
    ├──→ (×) ──→ b ──┘                 ├──→ (×) ──→ f
x ──┘                                  │
                          c ───────────┘

反向传播是**从输出往回走**，即**"最终结果 f 对每个中间变量、每个输入，变化率是多少？"**

核心工具就是**链式法则**：每往回走一层，就乘上这一层的局部导数





#### 从输入到 loss 的计算图

输入 X (1,2)
   │
   ├──(×W1ᵀ + b1)──→ Z1 (1,3)
   │                     │
   │                  (ReLU)
   │                     ↓
   │                   A1 (1,3)
   │                     │
   │              (×W2ᵀ + b2)──→ Z2 (1,1)
   │                              │
   │                          (Sigmoid)
   │                              ↓
   │                           Ŷ (1,1)
   │                              │
   │                          (BCE vs Y)
   │                              ↓
   │                           L (标量)
   ▼ 反向传播
dX ←── dZ1 ←── dA1 ←── dZ2 ←── dL=1
        │              │
        ├─ dW1(3,2)    ├─ dW2(1,3)
        └─ db1(3,)     └─ db2(1,)



- 前向传播时为什么要保存每一层的输入或输出？反向传播会在哪里用到它们？这和显存有什么关系？

反向传播求梯度时，**局部导数往往依赖前向时的值**，所以前向必须把这些值存下来

网络越深、batch 越大，保存的中间张量越多，显存占用越大

这就是为什么大模型要用**梯度检查点（gradient checkpointing）**：只存部分层，其余反向时重算，用时间换显存

- 链式法则怎样把损失函数的梯度传回第一层权重？请结合自己的计算图说明



链式法则：复合函数求导，导数相乘
计算图路径：
$X \rightarrow W_1 \rightarrow Z_1=XW_1+b_1 \rightarrow A_1=\sigma(Z_1)\rightarrow W_2\rightarrow Z_2=A_1W_2+b_2\rightarrow \hat Y \rightarrow Loss$

1. 先算最末端：\frac{dLoss}{d\hat Y}（损失对预测输出梯度）

2. 逐层往前求导：
$\frac{dLoss}{dZ_2}=\frac{dLoss}{d\hat Y}\cdot\frac{d\hat Y}{dZ_2}
\frac{dLoss}{dW_2}=A_1^\top \cdot \frac{dLoss}{dZ_2}
\frac{dLoss}{dA_1}=\frac{dLoss}{dZ_2}\cdot W_2^\top
\frac{dLoss}{dZ_1}=\frac{dLoss}{dA_1}\cdot \sigma'(Z_1)
\boldsymbol{\frac{dLoss}{dW_1}=X^\top \cdot \frac{dLoss}{dZ_1}}$

3. 一直反向传递，直到求出最第一层权重 W_1 的梯度







- 二元交叉熵与 Sigmoid 配合后，loss 对 logit 的梯度是什么？这个结果为什么方便反向传播？

设：
$\hat y=\sigma(z),\quad \sigma(z)=\frac{1}{1+e^{-z}}$
二元交叉熵损失：
$L = - \big[y\log\sigma(z)+(1-y)\log(1-\sigma(z))\big]$

求导化简：
$\boldsymbol{\frac{dL}{dz}=\sigma(z)-y=\hat y - y}$

为什么方便反向传播：
化简之后不再单独计算sigmoid导数，直接是预测值减标签，形式非常简洁，数值计算稳定，减少指数运算，代码写起来简单











- NumPy 梯度与 PyTorch 梯度出现差异时，你会怎样判断问题出在公式、矩阵转置、广播还是损失的求平均方式？

1. 损失求平均方式：
PyTorch默认 mean 求平均（除以样本总数）；numpy手写如果是 sum 求和，梯度会差一个倍数，数值整体放大缩小。最简单判断：看loss大小是否差N倍(N=样本数)

2. 矩阵转置问题：矩阵乘法维度不对，梯度形状不对，数值完全不对,检查  W @ X  和  X @ W ，以及 $dW=A_{prev}.T @ dZ$ 转置方向。

3. 广播问题：偏置 db 计算时广播错误，偏置梯度形状对不上，偏置梯度数值异常

4. 公式错误：形状没问题、倍数没问题，但每个元素都有固定差值，说明求导公式写错（比如BCE+sigmoid导数记错）





- 完成两次验证后，你现在怎样理解 `loss.backward()`？它替我们保存和完成了哪些工作？

前向阶段：自动记录计算图，保存所有需要用于求导的中间张量（就是前面说的每层输入输出）

调用 backward() 时

- 从 loss 张量开始，沿着计算图反向遍历

- 自动应用链式法则，逐层求导

- 把算出来的梯度存入每个张量的  .grad  属性（ w.grad 、 b.grad ）；

3. 完成的工作：自动构建计算图、保存中间结果、链式法则反向求导、把梯度写入参数 .grad ，不用手写反向传播代码







- 将你遇到的报错or不理解的点记录。

1. 一开始分不清  sum  和  mean ，numpy用sum，pytorch用mean，梯度差了样本倍数，导致梯度对比不一致；

2. 矩阵乘法转置搞反， A @ W  和  W @ A  维度报错，梯度形状不匹

3. 混淆  BCE  和  BCEWithLogitsLoss ，重复sigmoid导致求导错误

4. 一开始不理解：为什么前向要存中间值，显存占用高

5. 刚开始不明白  .grad  会累加梯度，不做清零，多次backward后梯度爆炸