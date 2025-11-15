import os
from tqdm import tqdm  # 进度条显示
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
from PIL import Image  # 图像处理
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import Dataset, DataLoader, random_split  # 数据集和数据加载
from torchvision import transforms  # 图像变换
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # 余弦退火学习率调度器
from transformers import SwinModel, BlipProcessor, BlipForConditionalGeneration  # 模型导入
import torch.nn.functional as F  # 函数式接口
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 学习率调度器
import time  # 时间测量
from sklearn.metrics import mean_squared_error  # 评估指标
from scipy.stats import spearmanr, pearsonr  # 相关系数
import matplotlib.pyplot as plt  # 可视化
from multiprocessing import freeze_support  # 多进程支持
import logging  # 日志记录
import requests  # HTTP请求
import json  # JSON处理

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multimodal_aesthetic_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. 增强的自定义数据集类 (MultimodalAestheticDataset)
# 该类负责加载图像数据和对应的美学评分，并处理损坏的图像
class MultimodalAestheticDataset(Dataset):
    """
    多模态美学数据集类，用于加载和预处理图像及其美学评分
    
    参数:
        csv_file (str): 包含图像文件名和评分的CSV文件路径
        img_dir (str): 图像文件夹路径
        transform (callable, optional): 应用于图像的转换
        skip_broken (bool): 是否跳过损坏的图像，默认为True
        caption_processor: BLIP处理器，用于生成图像描述
        caption_model: BLIP模型，用于生成图像描述
        device: 计算设备
    """
    def __init__(self, csv_file, img_dir, transform=None, skip_broken=True, 
                 caption_processor=None, caption_model=None, device='cpu'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.ToTensor()  # 确保至少有ToTensor转换
        self.skip_broken = skip_broken
        self.caption_processor = caption_processor
        self.caption_model = caption_model
        self.device = device

        if skip_broken:
            valid_indices = []
            for idx in tqdm(range(len(self.data)), desc="验证图像"):
                img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
                try:
                    with Image.open(img_name) as img:
                        valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"跳过损坏的图像 {img_name}: {e}")

            self.data = self.data.iloc[valid_indices].reset_index(drop=True)
            logger.info(f"有效图像数量: {len(self.data)}/{len(valid_indices) + (len(self.data) - len(valid_indices))}")

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (图像张量, 评分张量, 图像描述)
        """
        try:
            # 获取图像路径和分数
            img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
            score = self.data.iloc[idx, 1]

            # 打开并转换图像
            try:
                image = Image.open(img_name).convert('RGB')
                original_image = image.copy()  # 保存原始图像用于生成描述
            except Exception as e:
                logger.error(f"加载图像失败 {img_name}: {e}")
                # 如果图像加载失败，返回一个空白图像
                image = Image.new('RGB', (224, 224))
                original_image = image.copy()

            # 应用转换
            if self.transform:
                try:
                    transformed_image = self.transform(image)
                except Exception as e:
                    logger.error(f"转换图像失败 {img_name}: {e}")
                    # 如果转换失败，至少确保返回张量
                    transformed_image = transforms.ToTensor()(image)

            # 生成图像描述
            caption = ""
            if self.caption_processor is not None and self.caption_model is not None:
                try:
                    # 准备图像用于BLIP模型
                    inputs = self.caption_processor(original_image, return_tensors="pt").to(self.device)
                    # 生成图像描述
                    with torch.no_grad():
                        outputs = self.caption_model.generate(**inputs, max_length=50)
                    caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
                except Exception as e:
                    logger.error(f"生成图像描述失败 {img_name}: {e}")
                    caption = ""  # 如果生成失败，使用空字符串

            return transformed_image, torch.tensor(score, dtype=torch.float32), caption

        except Exception as e:
            logger.error(f"处理样本失败 {idx}: {e}")
            # 返回一个有效的默认值
            return torch.zeros((3, 224, 224)), torch.tensor(0.0), ""

# 2. 数据转换定义函数
# 为训练集和验证/测试集定义不同的数据增强策略
def get_transforms(mode='train'):
    """
    获取图像转换函数
    
    参数:
        mode (str): 'train'表示训练模式，使用数据增强；'val'或'test'表示验证/测试模式，只进行基本处理
        
    返回:
        transforms.Compose: 图像转换组合
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),  # 首先调整大小
            transforms.RandomCrop(224),     # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转±10度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度和饱和度
            transforms.ToTensor(),          # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用ImageNet预训练模型的标准化参数
        ])
    else:  # 'val' 或 'test'
        return transforms.Compose([
            transforms.Resize((256, 256)),  # 调整大小
            transforms.CenterCrop(224),    # 中心裁剪
            transforms.ToTensor(),         # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

# 3. 改进的模态专家混合模块
# 该模块实现了混合专家机制，允许模型学习多个专家网络并动态选择最相关的专家
class MOE(nn.Module):
    """
    模态专家混合(Mixture of Experts)模块
    
    参数:
        num_experts (int): 专家网络的数量
        visual_input_size (int): 视觉特征的维度
        text_input_size (int): 文本特征的维度
        expert_output_size (int): 每个专家网络输出的维度
        dropout_rate (float): Dropout比率，用于防止过拟合
    """
    def __init__(self, num_experts, visual_input_size, text_input_size, expert_output_size, dropout_rate=0.1):
        super(MOE, self).__init__()
        self.num_experts = num_experts
        
        # 特征投影层，将视觉和文本特征投影到相同的维度
        self.visual_projection = nn.Linear(visual_input_size, expert_output_size)
        self.text_projection = nn.Linear(text_input_size, expert_output_size)
        
        # 计算门控网络输入维度（视觉特征和文本特征的拼接）
        gate_input_size = expert_output_size * 2
        
        # 创建专家网络，每个专家由两个线性层、激活函数、Dropout和LayerNorm组成
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gate_input_size, expert_output_size),  # 第一个线性层
                nn.GELU(),  # GELU激活函数
                nn.Dropout(dropout_rate),  # Dropout防止过拟合
                nn.Linear(expert_output_size, expert_output_size),  # 第二个线性层
                nn.LayerNorm(expert_output_size)  # 层归一化
            ) for _ in range(num_experts)
        ])
        
        # 门控网络，用于决定每个专家的权重
        self.gate = nn.Linear(gate_input_size, num_experts)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, visual_features, text_features):
        """
        前向传播函数
        
        参数:
            visual_features (torch.Tensor): 视觉特征张量，形状为(batch_size, visual_input_size)
            text_features (torch.Tensor): 文本特征张量，形状为(batch_size, text_input_size)
            
        返回:
            torch.Tensor: 混合后的输出，形状为(batch_size, expert_output_size)
        """
        # 投影特征到相同维度
        projected_visual = self.visual_projection(visual_features)  # (batch_size, expert_output_size)
        projected_text = self.text_projection(text_features)  # (batch_size, expert_output_size)
        
        # 拼接特征作为MOE的输入
        combined_features = torch.cat([projected_visual, projected_text], dim=1)  # (batch_size, expert_output_size*2)
        
        # 计算门控权重
        gate_logits = self.gate(combined_features)  # (batch_size, num_experts)
        gate_probabilities = F.softmax(gate_logits, dim=1)  # (batch_size, num_experts)，使用softmax归一化权重
        gate_probabilities = self.dropout(gate_probabilities)  # 增加随机性

        # 通过每个专家网络处理特征
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](combined_features))  # (batch_size, expert_output_size)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, expert_output_size)

        # Gating - 使用爱因斯坦求和约定计算加权和
        gated_outputs = torch.einsum("beo,be->bo", expert_outputs,
                                     gate_probabilities)  # (batch_size, expert_output_size)
        return gated_outputs

# 4. 文本编码器模块
class TextEncoder(nn.Module):
    """
    文本编码器模块，用于处理BLIP生成的图像描述
    
    参数:
        hidden_size (int): 隐藏层大小
        output_size (int): 输出特征维度
        dropout_rate (float): Dropout比率
    """
    def __init__(self, hidden_size=768, output_size=512, dropout_rate=0.1):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act1 = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.act2 = nn.GELU()
        self.norm2 = nn.LayerNorm(output_size)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入文本特征
            
        返回:
            torch.Tensor: 处理后的文本特征
        """
        x = self.fc1(x)
        x = self.act1(x)
        x = self.norm1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.norm2(x)
        x = self.drop2(x)
        return x

# 5. 多模态美学质量评估模型
class MultimodalAestheticModel(nn.Module):
    """
    多模态美学质量评估模型，结合视觉和文本特征
    
    参数:
        num_experts (int): MOE模块中的专家数量，默认为6
        expert_output_size (int): 每个专家的输出维度，默认为512
        dropout_rate (float): Dropout比率，默认为0.1
    """
    def __init__(self, num_experts=6, expert_output_size=512, dropout_rate=0.1):
        super(MultimodalAestheticModel, self).__init__()
        # 视觉编码器 - 使用预训练的Swin Transformer
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.visual_feature_dim = self.swin.config.hidden_size

        # 文本编码器 - 使用BLIP模型
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.text_feature_dim = self.blip_model.config.text_config.hidden_size
        
        # 文本特征处理器
        self.text_encoder = TextEncoder(
            hidden_size=self.text_feature_dim,
            output_size=expert_output_size,
            dropout_rate=dropout_rate
        )

        # 冻结部分Swin Transformer层和BLIP模型
        for name, param in self.swin.named_parameters():
            if 'layer.0' in name or 'layer.1' in name:  # 冻结前两层
                param.requires_grad = False
                
        for param in self.blip_model.parameters():
            param.requires_grad = False  # 完全冻结BLIP模型

        # 多模态特征融合MOE门控网络
        self.moe_gating_network = MOE(
            num_experts=num_experts,
            visual_input_size=self.visual_feature_dim,
            text_input_size=self.text_feature_dim,
            expert_output_size=expert_output_size,
            dropout_rate=dropout_rate
        )
        
        # 特征融合层
        self.fusion_layer = nn.Linear(expert_output_size, expert_output_size)
        self.fusion_act = nn.GELU()
        self.fusion_norm = nn.LayerNorm(expert_output_size)
        self.fusion_drop = nn.Dropout(dropout_rate)

        # 评分预测层
        self.fc1 = nn.Linear(expert_output_size, 256)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 64)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(64)

        # 回归器，输出美学评分
        self.regressor = nn.Linear(64, 1)
        
        # 存储生成的图像描述，用于后续GPT增强
        self.generated_captions = {}

    def extract_text_features(self, images, batch_indices=None):
        """
        提取图像的文本描述特征
        
        参数:
            images (torch.Tensor): 输入图像
            batch_indices (list): 批次索引，用于存储生成的描述
            
        返回:
            tuple: (文本特征, 图像描述)
        """
        # 将图像转换为PIL格式以供BLIP处理
        batch_size = images.size(0)
        text_features = torch.zeros(batch_size, self.text_feature_dim).to(images.device)
        image_captions = ["" for _ in range(batch_size)]
        
        # 反归一化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        unnormalized_images = images * std + mean
        unnormalized_images = unnormalized_images.clamp(0, 1)
        
        for i in range(batch_size):
            # 转换为PIL图像
            img = transforms.ToPILImage()(unnormalized_images[i].cpu())
            
            # 使用BLIP生成描述并提取特征
            inputs = self.blip_processor(img, return_tensors="pt").to(images.device)
            with torch.no_grad():
                # 生成图像描述
                outputs = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                image_captions[i] = caption
                
                # 获取文本特征
                text_inputs = self.blip_processor.tokenizer(caption, return_tensors="pt").to(images.device)
                text_outputs = self.blip_model.text_encoder(**text_inputs, output_hidden_states=True)
                # 使用最后一层隐藏状态的平均值作为文本特征
                text_features[i] = text_outputs.hidden_states[-1].mean(dim=1).squeeze(0)
                
                # 如果提供了批次索引，存储生成的描述
                if batch_indices is not None:
                    self.generated_captions[batch_indices[i]] = caption
                
        return text_features, image_captions

    def forward(self, images, batch_indices=None):
        """
        前向传播函数
        
        参数:
            images (torch.Tensor): 输入图像张量，形状为(batch_size, 3, 224, 224)
            batch_indices (list): 批次索引，用于存储生成的描述
            
        返回:
            torch.Tensor: 预测的美学评分，形状为(batch_size, 1)
        """
        # 1. 提取视觉特征 - 视觉特征提取路径
        visual_outputs = self.swin(images)
        visual_features = visual_outputs.pooler_output  # (batch_size, visual_feature_dim)
        
        # 2. 提取文本特征 - 文本特征提取路径
        text_features, image_captions = self.extract_text_features(images, batch_indices)  # (batch_size, text_feature_dim)
        
        # 3. 多模态特征融合 - MOE门控网络
        # 将视觉特征和文本特征同时输入到MOE门控网络
        fused_features = self.moe_gating_network(visual_features, text_features)  # (batch_size, expert_output_size)
        
        # 4. 特征融合后处理
        fused_features = self.fusion_layer(fused_features)  # (batch_size, expert_output_size)
        fused_features = self.fusion_act(fused_features)
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.fusion_drop(fused_features)
        
        # 5. 预测评分 - 美学评分预测
        x = self.fc1(fused_features)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.norm1(x)
        
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.norm2(x)
        
        score = self.regressor(x)
        return score
    
    def get_image_description(self, idx):
        """
        获取指定索引的图像描述
        
        参数:
            idx: 图像索引
            
        返回:
            str: 图像描述
        """
        return self.generated_captions.get(idx, "")
    
    def generate_detailed_description(self, idx, score, api_key):
        """
        使用GPT生成详细的图像描述
        
        参数:
            idx: 图像索引
            score: 美学评分
            api_key: GPT API密钥
            
        返回:
            str: 详细的图像描述
        """
        caption = self.get_image_description(idx)
        if not caption:
            return "无法生成详细描述，未找到图像的基础描述。"
        
        # 调用GPT API生成详细描述
        try:
            url = "https://aibotpro.cn/v1/chat/completions"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt}的美学评分是 {score:.2f}。我对这张图片的文本描述是：'{caption}'。请基于这个美学评分和文本描述，更详细地描述这张图片，突出它的视觉元素和整体美感。"
                    }
                ],
                "stream": False
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
                "Accept": "*/*",
                "Host": "aibotpro.cn",
                "Connection": "keep-alive"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"无法生成详细描述。基本描述: {caption}. 美学评分: {score:.2f}. 错误: {str(e)}"

# 6. 早停类
# 用于监控验证损失并在性能不再提升时提前停止训练
class EarlyStopping:
    """
    早停机制，用于防止过拟合
    
    参数:
        patience (int): 容忍验证损失不改善的轮数
        min_delta (float): 最小改善量
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): 容忍验证损失不改善的轮数
            min_delta (float): 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        检查是否应该早停
        
        参数:
            val_loss (float): 当前验证损失
            
        返回:
            bool: 是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# 7. 预热调度器类
# 实现了学习率预热策略，在训练初期逐渐增加学习率
class WarmupScheduler:
    """
    学习率预热调度器，在训练初期逐渐增加学习率
    
    参数:
        optimizer: 优化器
        warmup_epochs (int): 预热轮数
        init_lr (float): 初始学习率
    """
    def __init__(self, optimizer, warmup_epochs, init_lr=1e-6):
        """
        Args:
            optimizer: 优化器
            warmup_epochs (int): 预热轮数
            init_lr (float): 初始学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.current_epoch = 0

        # 获取优化器中设置的学习率
        self.final_lr = optimizer.param_groups[0]['lr']

    def step(self):
        """
        更新学习率，每个epoch调用一次
        """
        self.current_epoch += 1
        progress = self.current_epoch / self.warmup_epochs
        # 线性增加学习率
        current_lr = self.init_lr + (self.final_lr - self.init_lr) * progress

        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

    def get_lr(self):
        """
        获取当前学习率
        
        返回:
            float: 当前学习率
        """
        return self.optimizer.param_groups[0]['lr']

# 8. GPT描述生成函数
def generate_gpt_description(aesthetic_score, image_caption, api_key, prompt="这张图片"):
    """
    使用GPT API生成详细的美学描述
    
    参数:
        aesthetic_score (float): 美学评分
        image_caption (str): BLIP生成的图像描述
        api_key (str): API密钥
        prompt (str): 提示词
        
    返回:
        str: 生成的详细描述
    """
    url = "https://aibotpro.cn/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}的美学评分是 {aesthetic_score:.2f}。我对这张图片的文本描述是：'{image_caption}'。请基于这个美学评分和文本描述，更详细地描述这张图片，突出它的视觉元素和整体美感。"
            }
        ],
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Host": "aibotpro.cn",
        "Connection": "keep-alive"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"GPT API调用失败: {e}")
        return f"无法生成详细描述。基本描述: {image_caption}. 美学评分: {aesthetic_score:.2f}"

# 9. 增强的训练和评估函数
# 包含了梯度累积、早停、学习率预热等高级训练技巧
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                num_epochs=20, save_path='./multimodal_aesthetic_model.pth'):
    """
    训练模型函数
    
    参数:
        model (nn.Module): 待训练的模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        num_epochs (int): 训练轮数
        save_path (str): 模型保存路径
        
    返回:
        nn.Module: 训练好的模型
    """
    start_time = time.time()
    best_val_loss = float('inf')
    best_val_corr = -1.0
    train_losses = []
    val_losses = []
    val_corrs = []

    # 添加早停
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

    # 添加学习率预热
    warmup_epochs = 3
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs)

    # 添加梯度累积
    gradient_accumulation_steps = 1

    print(f"开始训练，共{num_epochs}个epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        # 预热阶段使用预热调度器
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            print(f"预热阶段 {epoch+1}/{warmup_epochs}, 学习率: {warmup_scheduler.get_lr():.6f}")

        for i, (images, scores, _) in enumerate(train_loader):
            images = images.to(device)
            scores = scores.to(device).unsqueeze(1)

            # 梯度累积
            if i % gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, scores)
            
            # 梯度累积 - 缩放损失
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # 每累积一定步数后更新参数
            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * gradient_accumulation_steps
            batch_count += 1
            
            # 每100个batch打印一次
            if batch_count % 100 == 0:
                avg_loss = running_loss / batch_count
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{batch_count}/{len(train_loader)}], '
                      f'Average Loss: {avg_loss:.4f}')
        
        # 计算epoch平均损失
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        val_loss, val_corr = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_corrs.append(val_corr)
        
        # 预热阶段后使用学习率调度器
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印epoch状态
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]: '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Corr: {val_corr:.4f}, '
              f'LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_val_loss = val_loss
            torch.save(model, save_path)
            torch.save(model.state_dict(), f"{save_path.split('.')[0]}_weights.pth")
            print(f'\n新的最佳模型! Corr: {val_corr:.4f}, Loss: {val_loss:.4f}')
        
        # 检查是否早停
        if early_stopping(val_loss):
            print(f"\n早停触发，在epoch {epoch+1}停止训练")
            break
    
    # 训练完成统计
    total_time = time.time() - start_time
    print(f'\n训练完成，耗时 {total_time/60:.2f} 分钟')
    print(f'最佳验证相关系数: {best_val_corr:.4f}')
    print(f'最佳验证损失: {best_val_loss:.4f}')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_corrs, save_path.replace('.pth', '_curves.png'))
    
    return model

def evaluate_model(model, dataloader, criterion, device):
    """
    评估模型函数
    
    参数:
        model (nn.Module): 待评估的模型
        dataloader (DataLoader): 数据加载器
        criterion: 损失函数
        device: 计算设备
        
    返回:
        tuple: (平均损失, Spearman相关系数)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, scores, _ in dataloader:
            images = images.to(device)
            scores = scores.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, scores)
            total_loss += loss.item()
            
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(scores.squeeze().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    spearman_corr = spearmanr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0
    pearson_corr = pearsonr(all_preds, all_targets)[0] if len(all_preds) > 1 else 0
    
    logger.info(f"评估结果 - 损失: {avg_loss:.4f}, Spearman相关系数: {spearman_corr:.4f}, Pearson相关系数: {pearson_corr:.4f}")
    
    return avg_loss, spearman_corr

def plot_training_curves(train_losses, val_losses, val_corrs, save_path):
    """
    绘制训练曲线
    
    参数:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        val_corrs (list): 验证相关系数列表
        save_path (str): 图像保存路径
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制相关系数曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_corrs, 'g-', label='验证相关系数')
    plt.title('验证Spearman相关系数')
    plt.xlabel('Epochs')
    plt.ylabel('相关系数')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"训练曲线已保存到 {save_path}")

# 10. 主函数
if __name__ == "__main__":
    # 支持Windows多进程
    freeze_support()
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 数据路径
    train_csv = r'/content/extracted_labels/labels/merge/train.csv'  # 训练集CSV文件路径
    val_csv = r'/content/extracted_labels/labels/merge/test.csv'    # 验证/测试集CSV文件路径
    img_dir =  r'/content/extracted_TAD66K'            # 图像目录路径，请根据实际情况修改
    
    # 初始化BLIP模型用于生成图像描述
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    caption_model.eval()  # 设置为评估模式
    
    # 创建数据集
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = MultimodalAestheticDataset(
        csv_file=train_csv,
        img_dir=img_dir,
        transform=train_transform,
        caption_processor=caption_processor,
        caption_model=caption_model,
        device=device
    )
    
    val_dataset = MultimodalAestheticDataset(
        csv_file=val_csv,
        img_dir=img_dir,
        transform=val_transform,
        caption_processor=caption_processor,
        caption_model=caption_model,
        device=device
    )
    
    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 初始化模型
    model = MultimodalAestheticModel(num_experts=6, expert_output_size=512, dropout_rate=0.1).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 训练模型
    num_epochs = 30
    model_save_path = "multimodal_aesthetic_model.pth"
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        save_path=model_save_path
    )
    
    logger.info(f"模型训练完成，已保存到 {model_save_path}")
    
    # 测试单张图像
    test_image_path = "./test_image.jpg"  # 测试图像路径
    api_key = "sk-4245c0d9f27f1fd3c51c86b03b6cfe12"  # API密钥
    
    if os.path.exists(test_image_path):
        # 加载和预处理图像
        test_image = Image.open(test_image_path).convert('RGB')
        test_tensor = val_transform(test_image).unsqueeze(0).to(device)
        
        # 预测美学评分并获取BLIP生成的图像描述
        with torch.no_grad():
            # 使用模型前向传播获取评分
            score = model(test_tensor, batch_indices=[0])
            predicted_score = score.item()
            
            # 获取BLIP生成的图像描述
            caption = model.get_image_description(0)
            
            # 使用GPT生成详细描述
            detailed_description = model.generate_detailed_description(0, predicted_score, api_key)
        
        logger.info(f"测试图像: {test_image_path}")
        logger.info(f"预测评分: {predicted_score:.2f}")
        logger.info(f"BLIP描述: {caption}")
        logger.info(f"GPT详细描述: {detailed_description}")
        
        # 打印结果
        print("\n测试结果:")
        print(f"图像路径: {test_image_path}")
        print(f"美学评分: {predicted_score:.2f}")
        print(f"初步描述 (BLIP): {caption}")
        print(f"\n详细描述 (GPT-4o-mini):\n{detailed_description}")
    else:
        logger.warning(f"测试图像 {test_image_path} 不存在")