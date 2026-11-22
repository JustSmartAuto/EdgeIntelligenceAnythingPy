# EdgeIntelligenceAnythingPy

面向边缘智能视觉任务的开源项目合集。当前收录 **ClassifyAnythingPy**：使用合成图像训练 Vision Transformer（ViT），完成自定义图像分类，并支持 Hugging Face Hub 模型管理。

## ClassifyAnythingPy

`ClassifyAnythingPy/classify-anything-main.zip` 包含完整源码、Poetry 依赖锁文件、训练配置和使用说明。

- 基于 `google/vit-base-patch16-224` 微调图像分类模型
- 训练图像按类别存放在不同子目录中，无需手动标注
- 使用 Hugging Face Transformers、PyTorch 和 Datasets
- 支持准确率评估、TensorBoard 日志与训练指标可视化
- 支持单张图片预测和目录批量推理

### 快速开始

建议使用 Python 3.11，并先安装 [Poetry](https://python-poetry.org/docs/#installation)。

```bash
unzip ClassifyAnythingPy/classify-anything-main.zip
cd classify-anything-main
poetry install
```

在 `config.yaml` 中配置数据目录、基础模型和训练参数。训练数据结构如下：

```text
artificial_data/
├── cats/
│   ├── 001.jpg
│   └── 002.jpg
└── dogs/
    ├── 001.jpg
    └── 002.jpg
test_data/
├── cats/
└── dogs/
```

开始训练：

```bash
poetry run python src/classification_utils/fine_tuning_vit.py
```

模型默认保存到 `outputs/vit-finetuned`。如需推送到 Hugging Face Hub，请通过环境变量提供访问令牌和模型名称，并在 `config.yaml` 中启用 `push_to_hub`：

```bash
export HUGGINGFACE_TOKEN="your_token"
export HUB_MODEL_NAME="your_namespace/your_model"
```

Windows PowerShell 可使用 `$env:HUGGINGFACE_TOKEN="your_token"` 和 `$env:HUB_MODEL_NAME="your_namespace/your_model"`。不要将真实令牌写入源码或提交到仓库。

### 图像推理

训练完成后，可通过 `ImageClassifier` 加载本地模型或 Hub 模型：

```python
from src.inference import ImageClassifier

classifier = ImageClassifier(
    hub_model_name="outputs/vit-finetuned",
    base_model_name="google/vit-base-patch16-224",
)

label = classifier.predict_single_image("path/to/image.jpg")
print(label)
```

批量推理支持 PNG、JPG、JPEG 和 WebP 图片。首次下载模型或推送到 Hub 时需要网络连接；训练能否使用 GPU 取决于本机 PyTorch 与 CUDA 环境。

## 项目结构

```text
EdgeIntelligenceAnythingPy/
├── ClassifyAnythingPy/
│   └── classify-anything-main.zip
├── index.html
├── LICENSE
└── README.md
```

## 项目主页

`index.html` 是无需构建的静态介绍页，可直接在浏览器打开，也可用于 GitHub Pages。

## 许可证

本仓库采用 [MIT License](LICENSE)。第三方模型、数据集和依赖仍遵循各自许可证与使用条款。
