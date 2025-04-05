import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class SSMConfig:
    """ 配置模型参数类 """
    def __init__(self,
                 d_model=512,  # 模型隐藏层维度
                 d_state=16,   # 状态空间维度
                 d_conv=4,     # 卷积核尺寸
                 expand=2):    # 扩展因子
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

class Mamba(nn.Module):
    """ Mamba模型核心实现 """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 输入投影层
        self.in_proj = nn.Linear(config.d_model, config.d_model * config.expand)

        # 因果卷积层: 采用深度可分离卷积处理局部模式
        self.conv1d = nn.Conv1d(
            in_channels=config.d_model * config.expand,
            out_channels=config.d_model * config.expand,
            kernel_size=config.d_conv,
            groups=config.d_model * config.expand, # 深度可分离卷积节省参数
            padding=config.d_conv - 1  # 因果卷积需要左侧填充
        )

        # 状态空间模型参数初始化
        # 连续状态空间A，使用复数形式初始化实现高频保留
        A_real = torch.randn(config.d_state, config.d_state//2) * 0.02
        A_imag = torch.randn(config.d_state, config.d_state//2) * 0.02
        self.A = nn.Parameter(torch.view_as_complex(torch.stack([A_real, A_imag], dim=-1)))

        # 输入/输出投影矩阵(B, C)和跳跃连接D
        self.B = nn.Parameter(torch.randn(config.d_model * config.expand, config.d_state))
        self.C = nn.Parameter(torch.randn(config.d_model * config.expand, config.d_state))
        self.D = nn.Parameter(torch.ones(config.d_model)) # 跳跃连接初始化为1

        # 选择性机制投影层：动态生成Δ和调整参数
        # 输出维度：Δ (1) + A调整参数(d_state) + B调整参数(d_model*expand) + C调整参数(d_model*expand)
        self.s_proj = nn.Linear(
            config.d_model * config.expand,
            1 + config.d_state + 2 * config.d_model * config.expand
        )

        # 输出投影层：将扩展维度恢复为原始维度
        self.out_proj = nn.Linear(config.d_model * config.expand, config.d_model)

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def discretization(self, delta):
        """
        连续系统离散化过程
        公式: Ā = exp(Δ * A), B̄ = (Ā - I) * A⁻¹ B
        输入: 
            delta: (B, L) 时间步长参数（由输入决定）
        输出: 
            A_disc: (B, L, d_state, d_state) 离散化后的状态矩阵
            B_disc: (B, L, d_model*expand, d_state) 离散化后的输入矩阵
        """
        # 将Δ与状态矩阵A进行外积运算
        # delta:(B, L) -> (B, L, 1)     A:(d_state, d_state) -> (1, d_state, d_state)
        deltaA = delta.unsqueeze(-1) * self.A.unsqueeze(0) # (B, L, d_state, d_state)

        # 矩阵指数运算实现离散化
        A_disc = torch.matrix_exp(deltaA)

        # 计算A矩阵的逆
        inv_A = torch.linalg.inv(self.A.unsqueeze(0)) # (1, d_state, d_state)

        # 计算B的离散化项
        B_disc = (A_disc - torch.eye(self.config.d_state).to(A_disc)) @ inv_A @ self.B.T #(B, L, d_state, d_model*expand)

        return A_disc, B_disc.permute(0, 1, 3, 2) # (B, L, d_state, d_model*expand) -> (B, L, d_model*expand, d_state)
    

    def selective_scan(self, x, delta, A_disc, B_disc, C):
        """选择性扫描核心计算(使用Triton优化)
        输入: 
            x: 输入序列 (B, L, D*expand)
            delta: 时间步长参数 (B, L)
            A_disc: 离散化状态矩阵A (B, L, d_state, d_state) 
            B_disc: 离散化输入矩阵B (B, L, D*expand, d_state)
            C: 动态调整后的输出矩阵C (B, K, D*expand, d_state)
        输出: 
            y: 输出序列 (B, L, D*expand)
        """

        B, L, D = x.shape
        d_state = self.config.d_state

        # 初始化隐藏状态
        h = torch.zeros(B, D, d_state, device=x.device) # (B, D, d_state)
        outputs = []

        # Triton优化后的并行扫描内核
        # 每个线程块处理一个时间步的多个通道
        grid = (B * L, )
        selective_scan_kernel[grid](
            x, delta, A_disc, B_disc, C, self.D,
            y=outputs,
            d_model=D, d_state=d_state, seq_len=L,
            BLOCK_MODEL=16, BLOCK_STATE=8
        )

    def forward(self, x):
        """
        前向传播流程: 
        1. 输入投影和因果卷积
        2. 生成动态参数Δ, Ã, B̃, C̃
        3. 离散化状态空间模型
        4. 执行选择性扫描
        5. 输出投影
        """
        B, L ,D = x.shape  # 输入维度 (Batch, Length, Dim)
        config = self.config

        # 阶段1：输入投影和因果卷积 ------------------------------
        x = self.in_proj(x) # (B, L, D*expand)
        x = x.transpose(1, 2) # (B, D*expand, L)  适应Conv1d的通道优先格式
        x = self.conv1d(x)[..., :L] # 因果卷积(截断右侧填充部分)
        x = x.transpose(1, 2) # (B, L, D*expand)

        # 阶段2：动态参数生成 ------------------------------
        s = self.s_proj(x) # (B, L, 1 + d_state + 2*D*expand)
        delta = F.softplus(s[..., 0]) # Δ必须为正，使用softplus激活
        A_mod = torch.sigmoid(s[..., 1:1+config.d_state])  # A调整参数
        B_mod = s[..., 1+config.d_state:1+config.d_state+config.d_model*config.expand]
        C_mod = s[..., 1+config.d_state+config.d_model*config.expand:]

        # 阶段3：离散化系统 ------------------------------
        A_disc, B_disc = self.discretization(delta)
        A_disc = A_disc * A_mod.unsqueeze(-1)  # 应用选择性调整
        B_total = B_disc + B_mod.unsqueeze(-1)  # 静态B + 动态调整

        # 阶段4：选择性扫描 ------------------------------
        y = self.selective_scan(x, delta, A_disc, B_total, C_mod)

        # 阶段5：输出投影和残差连接 ------------------------------
        return x + self.out_proj(y)  # (B, L, D)

@triton.jit
def selective_scan_kernel(
    x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
    y_ptr,
    d_model, d_state, seq_len,
    BLOCK_MODEL: tl.constexpr, BLOCK_STATE:tl.constexpr
):
    """
    Triton实现并行扫描内核
    关键技术点: 
    - 分块处理: 将模型维度分为BLOCK_MODEL块, 状态维度分为BLOCK_STATE块
    - 共享内存: 利用快速共享内存缓存中间状态
    - 双缓冲: 预加载下一时间步的数据隐藏内存延迟
    """
    # 线程块索引计算
    pid = tl.program_id(0)
    batch = pid // seq_len
    time_step = pid % seq_len

    # 内存偏移计算
    off_A = batch * d_state + tl.arange(0, BLOCK_STATE)
    a = tl.load(A_ptr + off_A) # 加载状态矩阵A

    # 共享内存分配(存储隐藏状态)
    shmem = tl.static_shared_memory((BLOCK_MODEL, BLOCK_STATE), dtype=tl.float32)

    # 时间步迭代(并行前缀求和)
    for t in range(time_step + 1):
        # 加载当前时间步的输入和参数
        off_x = batch * seq_len * d_model + t * d_model + tl.arange(0, BLOCK_MODEL)
        x = tl.load(x_ptr + off_x)
        delta = tl.load(delta_ptr + batch * seq_len + t)

        # 动态调整参数(核心数学操作)
        a_t = a * tl.exp(delta * a) # 离散化公式的指数运算
        b = tl.load(B_ptr + off_x * d_state)
        c = tl.load(C_ptr + off_x * d_state)

        # 状态更新公式：h_t = Ā * h_{t-1} + B̄ * x_t
        shmem = a_t * shmem + x[:, None] * b[None, :]
    
    # 计算当前时间步的输出：y_t = C * h_t + D * x_t
    y = tl.dot(shmem, c) + D_ptr[0] * x

    # 存储结果
    tl.store(y_ptr + (batch * seq_len + time_step) * d_model, y)