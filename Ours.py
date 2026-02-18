import torch					
import torch.nn as nn					
					
# 1. 原有Residual_2D残差块（保持不变）					
class Residual_2D(nn.Module):					
"    def __init__(self, in_channels, out_channels, kernel_size, padding, batch_normal=False):"					
"        super(Residual_2D, self).__init__()"					
        self.batch_normal = batch_normal					
        					
        # 主分支：两层卷积 + 可选批归一化 + 激活					
"        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)"					
"        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)"					
        					
        # 可选批归一化层					
        if self.batch_normal:					
            self.bn1 = nn.BatchNorm2d(out_channels)					
            self.bn2 = nn.BatchNorm2d(out_channels)					
        					
        # 激活函数					
        self.relu = nn.ReLU(inplace=True)					
        					
        # 捷径分支：当输入输出通道不一致时，用1x1卷积调整通道数					
        self.shortcut = nn.Sequential()					
        if in_channels != out_channels:					
"            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0))"					
					
"    def forward(self, x):"					
        # 保存原始输入用于残差连接					
        residual = x					
        					
        # 主分支前向传播					
        out = self.conv1(x)					
        if self.batch_normal:					
            out = self.bn1(out)					
        out = self.relu(out)					
        					
        out = self.conv2(out)					
        if self.batch_normal:					
            out = self.bn2(out)					
        					
        # 捷径分支调整通道数（如需）					
        residual = self.shortcut(residual)					
        					
        # 残差连接：主分支 + 捷径分支					
        out += residual					
        out = self.relu(out)					
        					
        return out					
					
# 2. 新增：高低频分解模块					
class HighLowFreqDecompose(nn.Module):					
    """					
    高光谱图像高低频分解：					
    - 低频分支：通过平滑卷积提取结构/全局信息（大尺度、低变化）					
    - 高频分支：原信号 - 低频信号，提取细节/纹理信息（小尺度、高变化）					
    """					
"    def __init__(self, in_channels, kernel_size=3):"					
        super().__init__()					
        # 低频提取：分组卷积（保持通道独立性）+ 平均权重初始化（模拟均值滤波）					
        self.low_freq_conv = nn.Conv2d(					
"            in_channels=in_channels,"					
"            out_channels=in_channels,"					
"            kernel_size=kernel_size,"					
"            padding=kernel_size//2,  # 保持尺寸不变"					
"            groups=in_channels,      # 分组数=通道数，避免跨通道干扰"					
            bias=False					
        )					
        # 初始化卷积核为平均权重（均值滤波）					
"        nn.init.constant_(self.low_freq_conv.weight, 1.0/(kernel_size*kernel_size))"					
					
"    def forward(self, x):"					
"        # x: (batch, band, H, W)"					
        low_freq = self.low_freq_conv(x)  # 低频特征（结构信息）					
        high_freq = x - low_freq          # 高频特征（细节/纹理信息）					
"        return low_freq, high_freq"					
					
# 3. 新增：跨频信息交互模块					
class CrossFreqInteraction(nn.Module):					
    """					
    高低频特征交互：实现高频→低频、低频→高频的双向信息反馈					
    - 门控机制：自适应调整高低频特征的融合权重					
    - 特征融合：将交互后的高低频特征融合为统一维度					
    """					
"    def __init__(self, in_channels, hidden_dim=64):"					
        super().__init__()					
        # 低频→高频的门控（控制低频对高频的指导权重）					
        self.low2high_gate = nn.Sequential(					
"            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),"					
"            nn.ReLU(inplace=True),"					
"            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),"					
            nn.Sigmoid()  # 输出0-1权重					
        )					
        # 高频→低频的门控（控制高频对低频的指导权重）					
        self.high2low_gate = nn.Sequential(					
"            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),"					
"            nn.ReLU(inplace=True),"					
"            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),"					
            nn.Sigmoid()  # 输出0-1权重					
        )					
        # 融合交互后的高低频特征（通道数还原为输入维度）					
"        self.fusion_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)"					
					
"    def forward(self, low_feat, high_feat):"					
        # 1. 计算双向门控权重					
        low2high_weight = self.low2high_gate(low_feat)    # 低频指导高频的权重					
        high2low_weight = self.high2low_gate(high_feat)    # 高频指导低频的权重					
        					
        # 2. 跨频信息交互（加权融合）					
        high_feat_inter = high_feat * low2high_weight + low_feat * (1 - low2high_weight)					
        low_feat_inter = low_feat * high2low_weight + high_feat * (1 - high2low_weight)					
        					
        # 3. 融合交互后的高低频特征					
"        fused_feat = self.fusion_conv(torch.cat([low_feat_inter, high_feat_inter], dim=1))"					
        					
        return fused_feat					
					
# 4. 修改后的CDCNN网络（融入高低频分解+跨频交互）					We have uploaded the code when the paper accepted.
					
# 测试代码（输入输出与原网络完全一致）					
if __name__ == "__main__":					
"    # 输入shape: (batch_size, 1, band, H, W)"					
"    a = torch.randn(4, 1, 30, 13, 13)"					
    # 初始化模型：band=30，分类数=16					
"    model = CDCNN_network(30, 16)"					
    out = model(a)					
"    # 输出shape应为 (4, 16)（batch_size=4，16类）"					
"    print(""输出shape:"", out.shape)  # 预期输出：torch.Size([4, 16])"					
