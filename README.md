# EdgeIntelligenceAnythingPy

面向边缘智能视觉任务的开源项目合集，覆盖图像分类、视觉目标定位以及图像与视频分割：

- **ClassifyAnythingPy**：使用合成图像训练 Vision Transformer（ViT），完成自定义图像分类。
- **LocateAnythingPy**：基于 NVIDIA Eagle 与 LocateAnything，通过自然语言定位目标、文字和 GUI 元素。
- **SegmentAnythingPy**：基于 Meta SAM 3，通过文本或视觉提示分割图像与视频中的目标并进行跟踪。

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

## LocateAnythingPy

`LocateAnythingPy/Eagle-main.zip` 收录 NVIDIA Eagle 项目源码，其中 `Embodied` 提供 LocateAnything 模型。目录同时包含论文 `2605.27365v2.pdf` 和 arXiv 源文件。

- **并行边界框解码（PBD）**：将完整边界框或点作为原子单元，在一次前向传播中预测坐标。
- **通用视觉定位**：支持目标检测、指代表达定位、密集目标检测、OCR 文字定位、GUI 元素定位和点定位。
- **混合推理**：默认使用快速并行模式，在输出不可靠时回退到自回归模式。
- **高效适配**：提供 LoRA 视觉提示微调脚本与批量推理运行时。

解压并安装：

```bash
unzip LocateAnythingPy/Eagle-main.zip
cd Eagle-main/Embodied
pip install -e .
```

基础推理示例：

```python
from PIL import Image
from locateanything_worker import LocateAnythingWorker

worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
image = Image.open("example.jpg").convert("RGB")

print(worker.detect(image, ["person", "car", "bicycle"])["answer"])
print(worker.ground_multi(image, "people wearing red shirts")["answer"])
print(worker.detect_text(image)["answer"])
```

详细安装选项、GPU 运行时、训练和评估方法见压缩包中的 `Embodied/document/lam/README.md`。代码采用 Apache 2.0 许可证，模型权重采用 NVIDIA 模型许可证。

## SegmentAnythingPy

`SegmentAnythingPy/sam3-main.zip` 收录 Meta **SAM 3（Segment Anything with Concepts）**源码；目录同时包含论文 `2511.16719v2.pdf` 和 arXiv 源文件。

- **概念分割**：根据简短文本或图像示例，分割开放词汇概念的全部实例。
- **多种提示方式**：支持文本、点、边界框和掩码提示。
- **图像与视频**：统一完成目标检测、实例分割、视频跟踪和交互式修正。
- **丰富示例**：提供图像、视频、批量推理和评估 Notebook。

SAM 3 要求 Python 3.12+、PyTorch 2.7+、支持 CUDA 的 GPU 和 CUDA 12.6+。安装示例：

```bash
unzip SegmentAnythingPy/sam3-main.zip
cd sam3-main
conda create -n sam3 python=3.12
conda activate sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

下载模型前，需要在 [Hugging Face 模型页面](https://huggingface.co/facebook/sam3)申请访问权限并运行 `hf auth login`。图像文本提示推理示例：

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
image = Image.open("example.jpg")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="person")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```

SAM 3 使用 SAM License，具体条款见压缩包内的 `LICENSE`。

## 项目结构

```text
EdgeIntelligenceAnythingPy/
├── ClassifyAnythingPy/
│   └── classify-anything-main.zip
├── LocateAnythingPy/
│   ├── Eagle-main.zip
│   ├── 2605.27365v2.pdf
│   └── arXiv-2605.27365v2.tar.gz
├── SegmentAnythingPy/
│   ├── sam3-main.zip
│   ├── 2511.16719v2.pdf
│   └── arXiv-2511.16719v2.tar.gz
├── index.html
├── LICENSE
└── README.md
```

## 项目主页

`index.html` 是无需构建的静态介绍页，可直接在浏览器打开，也可用于 GitHub Pages。

## 许可证

本仓库采用 [MIT License](LICENSE)。第三方模型、数据集和依赖仍遵循各自许可证与使用条款。
