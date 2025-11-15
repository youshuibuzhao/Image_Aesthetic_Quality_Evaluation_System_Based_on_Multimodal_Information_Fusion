
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from transformers import SwinModel, BlipProcessor, BlipForConditionalGeneration
import torch.nn.functional as F
import requests
import json
import os

# ============= Flask应用初始化 =============
app = Flask(__name__)
CORS(app)  # 启用跨域资源共享

# 检测并设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============= 图像预处理 =============
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============= 1. MOE模块（多模态双输入版本）=============
class MOE(nn.Module):
    """
    模态专家混合(Mixture of Experts)模块

    ✅ 关键特性：接收视觉和文本双输入，进行多模态融合

    参数:
        num_experts (int): 专家网络的数量
        visual_input_size (int): 视觉特征的维度（Swin输出：768）
        text_input_size (int): 文本特征的维度（BLIP输出：768）
        expert_output_size (int): 每个专家网络输出的维度（512）
        dropout_rate (float): Dropout比率，用于防止过拟合
    """
    def __init__(self, num_experts, visual_input_size, text_input_size, expert_output_size, dropout_rate=0.1):
        super(MOE, self).__init__()
        self.num_experts = num_experts

        # ✅ 特征投影层，将视觉和文本特征投影到相同的维度
        self.visual_projection = nn.Linear(visual_input_size, expert_output_size)
        self.text_projection = nn.Linear(text_input_size, expert_output_size)

        # ✅ 计算门控网络输入维度（视觉特征和文本特征的拼接）
        gate_input_size = expert_output_size * 2  # 512 * 2 = 1024

        # ✅ 创建专家网络，每个专家由两个线性层、激活函数、Dropout和LayerNorm组成
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gate_input_size, expert_output_size),  # 第一个线性层
                nn.GELU(),  # GELU激活函数
                nn.Dropout(dropout_rate),  # Dropout防止过拟合
                nn.Linear(expert_output_size, expert_output_size),  # 第二个线性层
                nn.LayerNorm(expert_output_size)  # 层归一化
            ) for _ in range(num_experts)
        ])

        # ✅ 门控网络，用于决定每个专家的权重
        self.gate = nn.Linear(gate_input_size, num_experts)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, visual_features, text_features):
        """
        ✅ 前向传播函数（双输入）

        参数:
            visual_features (torch.Tensor): 视觉特征张量，形状为(batch_size, 768)
            text_features (torch.Tensor): 文本特征张量，形状为(batch_size, 768)

        返回:
            torch.Tensor: 融合后的输出，形状为(batch_size, 512)
        """
        # ✅ 投影特征到相同维度
        projected_visual = self.visual_projection(visual_features)  # (batch, 512)
        projected_text = self.text_projection(text_features)        # (batch, 512)

        # ✅ 拼接特征作为MOE的输入
        combined_features = torch.cat([projected_visual, projected_text], dim=1)  # (batch, 1024)

        # 计算门控权重
        gate_logits = self.gate(combined_features)  # (batch, 6)
        gate_probabilities = F.softmax(gate_logits, dim=1)  # 使用softmax归一化权重
        gate_probabilities = self.dropout(gate_probabilities)  # 增加随机性

        # 通过每个专家网络处理特征
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](combined_features))  # (batch, 512)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, 6, 512)

        # Gating - 使用爱因斯坦求和约定计算加权和
        gated_outputs = torch.einsum("beo,be->bo", expert_outputs, gate_probabilities)  # (batch, 512)
        return gated_outputs


# ============= 2. 文本编码器模块 =============
class TextEncoder(nn.Module):
    """
    文本编码器模块，用于处理BLIP生成的图像描述

    参数:
        hidden_size (int): 隐藏层大小（768）
        output_size (int): 输出特征维度（512）
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


# ============= 3. 多模态美学质量评估模型 =============
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

        # ✅ 视觉编码器 - 使用预训练的Swin Transformer
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.visual_feature_dim = self.swin.config.hidden_size  # 768

        # ✅ 文本编码器 - 使用BLIP模型
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.text_feature_dim = self.blip_model.config.text_config.hidden_size  # 768

        # ✅ 文本特征处理器
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

        # ✅ 多模态特征融合MOE门控网络（双输入）
        self.moe_gating_network = MOE(
            num_experts=num_experts,
            visual_input_size=self.visual_feature_dim,  # 768
            text_input_size=self.text_feature_dim,       # 768
            expert_output_size=expert_output_size,       # 512
            dropout_rate=dropout_rate
        )

        # ✅ 特征融合层
        self.fusion_layer = nn.Linear(expert_output_size, expert_output_size)
        self.fusion_act = nn.GELU()
        self.fusion_norm = nn.LayerNorm(expert_output_size)
        self.fusion_drop = nn.Dropout(dropout_rate)

        # ✅ 评分预测层
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

        # 存储生成的图像描述（可选）
        self.generated_captions = {}

    def extract_text_features(self, images, batch_indices=None):
        """
        ✅ 提取图像的文本描述特征

        参数:
            images (torch.Tensor): 输入图像张量，形状为(batch_size, 3, 224, 224)
            batch_indices (list): 批次索引，用于存储生成的描述（可选）

        返回:
            tuple: (文本特征张量(batch_size, 768), 图像描述列表)
        """
        batch_size = images.size(0)
        text_features = torch.zeros(batch_size, self.text_feature_dim).to(images.device)
        image_captions = ["" for _ in range(batch_size)]

        # 反归一化图像（从ImageNet标准恢复到[0, 1]）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        unnormalized_images = images * std + mean
        unnormalized_images = unnormalized_images.clamp(0, 1)

        for i in range(batch_size):
            # 转换为PIL图像
            img = transforms.ToPILImage()(unnormalized_images[i].cpu())

            # ✅ 使用BLIP生成描述并提取特征
            inputs = self.blip_processor(img, return_tensors="pt").to(images.device)
            with torch.no_grad():
                # 生成图像描述
                outputs = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
                image_captions[i] = caption

                # ✅ 关键：提取文本特征
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
        ✅ 前向传播函数（多模态融合）

        参数:
            images (torch.Tensor): 输入图像张量，形状为(batch_size, 3, 224, 224)
            batch_indices (list): 批次索引，用于存储生成的描述（可选）

        返回:
            torch.Tensor: 预测的美学评分，形状为(batch_size, 1)
        """
        # 1. ✅ 提取视觉特征 - 视觉特征提取路径
        visual_outputs = self.swin(images)
        visual_features = visual_outputs.pooler_output  # (batch_size, 768)

        # 2. ✅ 提取文本特征 - 文本特征提取路径
        text_features, image_captions = self.extract_text_features(images, batch_indices)  # (batch_size, 768)

        # 3. ✅ 多模态特征融合 - MOE门控网络（双输入！）
        # 将视觉特征和文本特征同时输入到MOE门控网络
        fused_features = self.moe_gating_network(visual_features, text_features)  # (batch_size, 512)

        # 4. ✅ 特征融合后处理
        fused_features = self.fusion_layer(fused_features)  # (batch_size, 512)
        fused_features = self.fusion_act(fused_features)
        fused_features = self.fusion_norm(fused_features)
        fused_features = self.fusion_drop(fused_features)

        # 5. ✅ 预测评分 - 美学评分预测
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


# ============= 加载模型 =============
print("\n" + "="*70)
print("正在初始化多模态美学评估模型...")
print("="*70)

aesthetic_model = MultimodalAestheticModel(
    num_experts=6,
    expert_output_size=512,
    dropout_rate=0.1
).to(device)

# ✅ 加载多模态训练权重
# 注意：必须使用multimodal_train.py训练的权重，不能使用单模态的version4_model.pth
model_save_path = os.path.join(os.path.dirname(__file__), '..', 'multimodal_aesthetic_model.pth')

if os.path.exists(model_save_path):
    try:
        loaded_model = torch.load(model_save_path, map_location=device, weights_only=False)

        if isinstance(loaded_model, dict):
            aesthetic_model.load_state_dict(loaded_model)
        else:
            aesthetic_model.load_state_dict(loaded_model.state_dict())

        print(f"✅ 成功加载多模态模型权重: {model_save_path}")
    except Exception as e:
        print(f"❌ 加载模型权重失败: {e}")
        print("   请确保权重文件是使用multimodal_train.py训练的多模态模型")
else:
    print(f"⚠️  警告: 未找到模型权重文件")
    print(f"   预期路径: {model_save_path}")
    print("   请确保使用 multimodal_train.py 训练模型并保存权重")
    print("   或者调整 model_save_path 指向正确的权重文件")

aesthetic_model.eval()

print("✅ 模型已设置为评估模式")
print("="*70)


# ============= API相关配置 =============
API_KEY = "sk-4245c0d9f27f1fd3c51c86b03b6cfe12"


def generate_description_api(aesthetic_score, image_caption, api_key, prompt="这张图片"):
    """
    使用外部API生成详细美学描述

    参数:
        aesthetic_score: 美学评分
        image_caption: BLIP生成的图像描述
        api_key: API密钥
        prompt: 提示词前缀

    返回:
        str: 生成的详细描述
    """
    url = "https://aibotpro.cn/v1/chat/completions"
    payload = {
        "model": "chatgpt-4o-latest",
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
        response.raise_for_status()
        response_json = response.json()
        if 'choices' in response_json and response_json['choices']:
            return response_json['choices'][0]['message']['content']
        else:
            return "Error: API 没有生成文本。"
    except requests.exceptions.RequestException as e:
        return f"API 请求错误: {e}"
    except json.JSONDecodeError:
        return "错误: 无法解析 API 响应为 JSON。"


# ============= Flask路由 =============

@app.route('/test', methods=['GET'])
def test():
    """测试路由 - 验证服务是否正常运行"""
    return jsonify({
        'message': '✅ 多模态美学评价系统运行正常！',
        'model': 'MultimodalAestheticModel',
        'architecture': {
            'visual_encoder': 'Swin Transformer (Swin-Tiny)',
            'text_encoder': 'BLIP (Salesforce/blip-image-captioning-base)',
            'fusion_method': 'MOE (Mixture of Experts, 6 experts)',
            'fusion_type': 'Dual-input (Visual + Text)',
        },
        'features': [
            '视觉特征提取（Swin Transformer）',
            '文本特征提取（BLIP）',
            '多模态特征融合（MOE双输入）',
            '美学评分预测'
        ],
        'status': 'ready'
    }), 200


@app.route('/process_image', methods=['POST'])
def process_image():
    """
    处理上传的图像并返回多模态美学评估结果

    请求:
        - image: 图像文件（multipart/form-data）

    返回:
        - aesthetic_score: 美学评分（基于视觉+文本特征融合）
        - image_caption: BLIP生成的图像描述
        - api_description: GPT生成的详细美学描述
        - model_info: 模型信息
    """
    if 'image' not in request.files:
        return jsonify({'error': '请求中没有图像文件'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择图像'}), 400

    try:
        # 打开并预处理图像
        img = Image.open(file.stream).convert('RGB')
        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # ✅ 使用多模态模型获取评分
            # forward()内部会自动：
            # 1. 提取视觉特征（Swin）
            # 2. 提取文本特征（BLIP）
            # 3. 双输入融合（MOE）
            aesthetic_score_tensor = aesthetic_model(image_tensor)
            aesthetic_score = aesthetic_score_tensor.item()

            # 单独生成图像描述用于展示
            inputs_caption = aesthetic_model.blip_processor(images=img, return_tensors="pt").to(device)
            outputs_caption = aesthetic_model.blip_model.generate(**inputs_caption, max_length=50)
            image_caption = aesthetic_model.blip_processor.decode(outputs_caption[0], skip_special_tokens=True)

            # 调用API生成详细描述
            api_description = generate_description_api(aesthetic_score, image_caption, API_KEY)

        return jsonify({
            'aesthetic_score': f"{aesthetic_score:.2f}",
            'image_caption': image_caption,
            'api_description': api_description,
            'model_info': {
                'type': 'multimodal',
                'visual_encoder': 'Swin-Tiny',
                'text_encoder': 'BLIP',
                'fusion_method': 'MOE (6 experts)',
                'fusion_inputs': 'Visual features + Text features',
                'note': '✅ 真正的多模态融合（符合论文理论）'
            }
        }), 200

    except Exception as e:
        print(f"处理图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': '处理图像时出错', 'details': str(e)}), 500


# ============= 启动服务 =============
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎨 多模态图像美学评价系统")
    print("="*70)
    print(f"📱 设备: {device}")
    print(f"🤖 模型: MultimodalAestheticModel")
    print(f"👁️  视觉编码器: Swin Transformer (Swin-Tiny)")
    print(f"📝 文本编码器: BLIP (Salesforce/blip-image-captioning-base)")
    print(f"🔀 融合方法: MOE (6 experts, dual-input)")
    print(f"✅ 多模态融合: 视觉特征 + 文本特征")
    print("="*70)
    print("📡 API端点:")
    print("   GET  /test           - 测试服务状态")
    print("   POST /process_image  - 评估图像美学")
    print("="*70)
    print("🚀 服务启动中...")
    print("="*70 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')
