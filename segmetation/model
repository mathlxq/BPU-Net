import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightModule(nn.Module):
    def __init__(self, feature_dim=64, temperature_init=2.0, bias_init=-0.5):
        super(DynamicWeightModule, self).__init__()
        # 可学习的质量指标权重参数
        self.alpha = nn.Parameter(torch.tensor(1.0 / 3), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1.0 / 3), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(1.0 / 3), requires_grad=True)
        # 温度系数与偏置项
        self.temperature = temperature_init
        self.bias = nn.Parameter(torch.tensor(bias_init), requires_grad=True)
        # Sobel算子（用于边缘清晰度计算）
        self.sobel_x = nn.Parameter(torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32),
                                    requires_grad=False)
        self.sobel_y = nn.Parameter(torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32),
                                    requires_grad=False)
        # 特征维度适配卷积
        self.adapt_conv = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, stride=1, padding=0)

    def normalize(self, x):
        """将张量归一化到[0,1]区间"""
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def compute_edge_sharpness(self, x):
        """计算边缘清晰度（ES）"""
        b, c, h, w = x.shape
        sobel_x = self.sobel_x.repeat(1, c, 1, 1)
        sobel_y = self.sobel_y.repeat(1, c, 1, 1)
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        es = torch.mean(grad_mag, dim=[2, 3])
        es = torch.mean(es, dim=1)
        return self.normalize(es)

    def compute_snr(self, x):
        """计算信噪比（SNR）"""
        b, c, h, w = x.shape
        signal = F.gaussian_blur(x, kernel_size=3, sigma=1.0)
        noise = x - signal
        signal_energy = torch.sum(signal ** 2, dim=[1, 2, 3])
        noise_energy = torch.sum(noise ** 2, dim=[1, 2, 3])
        snr = 10 * torch.log10(signal_energy / (noise_energy + 1e-8))
        return self.normalize(snr)

    def compute_feature_consistency(self, x_prev, x_curr):
        """计算特征一致性（FC）"""
        b, c, h, w = x_prev.shape
        mu_prev = F.avg_pool2d(x_prev, kernel_size=3, stride=1, padding=1)
        mu_curr = F.avg_pool2d(x_curr, kernel_size=3, stride=1, padding=1)
        sigma_prev = F.avg_pool2d(x_prev ** 2, kernel_size=3, stride=1, padding=1) - mu_prev ** 2
        sigma_curr = F.avg_pool2d(x_curr ** 2, kernel_size=3, stride=1, padding=1) - mu_curr ** 2
        sigma_prev_curr = F.avg_pool2d(x_prev * x_curr, kernel_size=3, stride=1, padding=1) - mu_prev * mu_curr
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        ssim = ((2 * mu_prev * mu_curr + C1) * (2 * sigma_prev_curr + C2)) / \
               ((mu_prev ** 2 + mu_curr ** 2 + C1) * (sigma_prev + sigma_curr + C2 + 1e-8))
        fc = torch.mean(ssim, dim=[1, 2, 3])
        return fc

    def forward(self, x_prev, x_curr):
        """前向传播：计算动态权重并融合特征"""
        # 计算质量指标
        es = self.compute_edge_sharpness(x_prev)
        snr = self.compute_snr(x_prev)
        fc = self.compute_feature_consistency(x_prev, x_curr)
        # 归一化指标权重
        weights = torch.softmax(torch.stack([self.alpha, self.beta, self.gamma]), dim=0)
        alpha_norm, beta_norm, gamma_norm = weights[0], weights[1], weights[2]
        # 计算综合质量得分与动态权重
        quality = alpha_norm * es + beta_norm * snr + gamma_norm * fc
        weight = torch.sigmoid(quality * self.temperature + self.bias)
        weight = torch.clamp(weight, min=0.1, max=0.95)
        weight = weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # 特征融合
        F_fuse = (1 - weight) * x_curr + weight * x_prev
        F_fuse = self.adapt_conv(torch.cat([F_fuse, x_curr], dim=1))
        F_fuse = F.relu(F_fuse)
        return F_fuse, weight.squeeze()


class BP_UNet(nn.Module):
    def __init__(self, num_channels=2, num_classes=2, multi_layer=True):
        super(BP_UNet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]
        self.multi_layer = multi_layer
        print('multi_layer ', self.multi_layer)
        self.down1 = Conv3x3(num_channels, num_feat[0])
        if self.multi_layer:
            addition = 1
        else:
            addition = 0
        self.down2 = Conv3x3(num_feat[0] + addition, num_feat[1])
        self.down3 = Conv3x3(num_feat[1] + addition, num_feat[2])
        self.down4 = Conv3x3(num_feat[2] + addition, num_feat[3])
        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[3], num_feat[4]))
        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3])
        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])
        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])
        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])
        self.final = nn.Sequential(nn.Conv2d(num_feat[0], num_classes, kernel_size=1))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=2))
        self.downconv1_1 = Conv3x3(num_channels, num_feat[0])
        # 连接
        self.down2_1 = DownConcat(num_feat[0], num_feat[1])
        self.downconv2_1 = Conv3x3(num_feat[0] + num_feat[1], num_feat[1])
        # 连接
        self.down3_1 = DownConcat(num_feat[1], num_feat[2])
        self.downconv3_1 = Conv3x3(num_feat[1] + num_feat[2], num_feat[2])
        # 连接
        self.down4_1 = DownConcat(num_feat[2], num_feat[3])
        self.downconv4_1 = Conv3x3(num_feat[2] + num_feat[3], num_feat[3])
        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2), Conv3x3(num_feat[3], num_feat[4]))
        self.up1_1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1_1 = Conv3x3(num_feat[4], num_feat[3])
        self.up2_1 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2_1 = Conv3x3(num_feat[3], num_feat[2])
        self.up3_1 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3_1 = Conv3x3(num_feat[2], num_feat[1])
        self.up4_1 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4_1 = Conv3x3(num_feat[1], num_feat[0])
        self.final = nn.Sequential(nn.Conv2d(num_feat[0], num_classes, kernel_size=1))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=2))

        # 新增动态权重模块
        self.dynamic_weight_module = DynamicWeightModule(feature_dim=64)
        # 特征提取辅助卷积（用于从输入图像中提取特征用于动态权重计算）
        self.feat_extract = Conv3x3(num_channels, 64)

    def forward(self, inputs, last):
        # 新增：动态权重特征融合
        # 提取当前输入和前一切片的特征
        curr_feat = self.feat_extract(inputs)
        # 前一切片last适配为64通道特征
        last_feat = F.conv2d(last, torch.ones(1, 1, 1, 1, device=last.device), stride=1, padding=0)
        last_feat = last_feat.repeat(1, 64, 1, 1)  # 扩展到64通道
        # 动态权重融合
        fused_feat, _ = self.dynamic_weight_module(last_feat, curr_feat)

        # Multi-level fusion（原有逻辑，将融合后的特征加入输入）
        inputs = torch.cat([inputs, fused_feat[:, :1, :, :]], 1)  # 取单通道与原输入拼接
        down1_feat = self.down1(inputs)
        if self.multi_layer:
            down2_last = self.pool(last)
            down2_pool = torch.cat([self.pool(down1_feat), down2_last], 1)
            down2_feat = self.down2(down2_pool)
            down3_last = self.pool(down2_last)
            down3_pool = torch.cat([self.pool(down2_feat), down3_last], 1)
            down3_feat = self.down3(down3_pool)
            down4_last = self.pool(down3_last)
            down4_pool = torch.cat([self.pool(down3_feat), down4_last], 1)
            down4_feat = self.down4(down4_pool)
        else:
            down2_feat = self.down2(self.pool(down1_feat))
            down3_feat = self.down3(self.pool(down2_feat))
            down4_feat = self.down4(self.pool(down3_feat))
        bottom_feat = self.bottom(down4_feat)
        up1_feat = self.up1(bottom_feat, down4_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down3_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down2_feat)
        up3_feat = self.upconv3(up3_feat)
        up4_feat = self.up4(up3_feat, down1_feat)
        up4_feat = self.upconv4(up4_feat)
        outputs = self.final(up4_feat)

        # 反向传播路径（原有逻辑）
        down1_feat_1 = self.downconv1_1(outputs)
        down2_last_1 = up3_feat
        down2_pool_1 = torch.cat([self.pool(down1_feat_1), down2_last_1], 1)
        down2_feat_1 = self.downconv2_1(down2_pool_1)
        down3_last_1 = up2_feat
        down3_pool_1 = torch.cat([self.pool(down2_feat_1), down3_last_1], 1)
        down3_feat_1 = self.downconv3_1(down3_pool_1)
        down4_last_1 = up1_feat
        down4_pool_1 = torch.cat([self.pool(down3_feat_1), down4_last_1], 1)
        down4_feat_1 = self.downconv4_1(down4_pool_1)
        bottom_feat_1 = self.bottom(down4_feat_1)
        up1_feat_1 = self.up1(bottom_feat_1, down4_feat_1)
        up1_feat_1 = self.upconv1_1(up1_feat_1)
        up2_feat_1 = self.up2(up1_feat_1, down3_feat_1)
        up2_feat_1 = self.upconv2_1(up2_feat_1)
        up3_feat_1 = self.up3(up2_feat_1, down2_feat_1)
        up3_feat_1 = self.upconv3_1(up3_feat_1)
        up4_feat_1 = self.up4(up3_feat_1, down1_feat_1)
        up4_feat_1 = self.upconv4_1(up4_feat_1)
        outputs_1 = self.final(up4_feat_1)

        return outputs_1


class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


class DownConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DownConcat, self).__init__()
        self.conv = nn.Conv2d(in_feat,
                              out_feat,
                              kernel_size=2,
                              stride=2)

    def forward(self, inputs, up_outputs):
        outputs = self.conv(inputs)
        out = torch.cat([up_outputs, outputs], 1)
        return out
