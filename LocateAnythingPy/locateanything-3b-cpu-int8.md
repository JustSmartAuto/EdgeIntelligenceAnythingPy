# LocateAnything-3B 在工控机上的 CPU 推理与 int8 量化实践

> 记录时间：2026-11-23　执行机器：OptPC（192.168.1.38）　结论：**能跑通，但效果与速度都不满足产线要求**

## 1. 结论摘要

| 项目 | 结果 |
| --- | --- |
| 模型 | `D:\JustSmart\LocateAnything-3B`（HF 快照，97 个文件，7.26 GB，3.518 B 参数，bf16） |
| 硬件 | i9-14900K / 24 线程 / 32 GB 内存 / **无 NVIDIA GPU**（Intel UHD 770） |
| 能否 CPU 运行 | 能，强制 `sdpa` 注意力后端 |
| fp32 单张推理 | 71.97 s（1022×672，prompt 927 tokens，输出上限 256 tokens） |
| int8 动态量化 | 量化耗时 126 s，推理 **36.4–36.75 s**，加速 **≈1.96×** |
| 量化产物 | `D:\JustSmartAuto\_ocl\locateanything3b_int8_dynamic.pt`，4.79 GB |
| 精度表现 | 本产线近景图（prompt `bolt`）输出完全退化：int8 为 43 个左边缘细长条，fp32 为覆盖整图的大框 |
| 生产可用性 | **不可用**：速度远低于节拍要求，且场景/提示词完全不适配 |

## 2. 环境准备

### 2.1 Python 与离线 wheel

- 解释器：`C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe`（3.13.9）
- 离线 wheel 包：`D:\安装包\wheels_pytorch_jax`（41 个 wheel，0.34 GB，全部 cp313/py3）

```powershell
$py -m pip install --no-index --find-links <wheel目录> --upgrade <全部 whl 路径>
```

安装后关键版本：`torch 2.12.0+cpu`、`torchvision 0.27.0+cpu`、`torchaudio 2.11.0+cpu`、`jax/jaxlib 0.10.1`、`numpy 2.4.6`、`scipy 1.17.1`、`opencv-python 4.13.0.92`、`ultralytics 8.4.67`。

> 注意：该 wheel 集里**没有** transformers / tokenizers / safetensors / huggingface_hub / accelerate，需要联网补装。

### 2.2 额外依赖

```powershell
$py -m pip install transformers==4.57.1 accelerate safetensors sentencepiece einops tqdm regex protobuf peft decord lmdb
```

- `decord`、`lmdb`：`AutoProcessor` 在 import 阶段硬依赖，缺失会直接 `ModuleNotFoundError`
- `transformers` 必须锁 **4.57.1**，原因见 4.1

## 3. 运行与量化方案

### 3.1 加载（fp32）

模型权重是 bf16，但动态量化只支持 fp32/fp16 的 Linear，因此统一以 fp32 载入：

```python
import torch                      # 必须最先导入，见 4.2
torch.set_num_threads(24)

from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig

P = r"D:\JustSmart\LocateAnything-3B"
tok  = AutoTokenizer.from_pretrained(P, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(P, trust_remote_code=True)
cfg  = AutoConfig.from_pretrained(P, trust_remote_code=True)
cfg._attn_implementation = "sdpa"   # 默认 magi 需要 CUDA

model = AutoModel.from_pretrained(
    P, config=cfg, torch_dtype=torch.float32, trust_remote_code=True,
    low_cpu_mem_usage=True, attn_implementation="sdpa")
model.eval()
```

### 3.2 int8 动态量化

```python
model = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
```

- 只量化 Linear（LLM 与 ViT 的注意力/FFN），Embedding、lm_head、Conv 保持 fp32
- torch 2.10+ 已将 `torch.ao.quantization` 标记为 deprecated，后续应迁移到 `torchao` 的 `quantize_` API，目前 2.12 仍可正常工作（仅告警）

### 3.3 推理调用

```python
img = Image.open(IMG).convert("RGB")
s = min(1.0, 1024.0 / max(img.size))
img = img.resize((int(img.size[0]*s)//14*14, int(img.size[1]*s)//14*14))  # 14 的倍数

msgs = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text",  "text": "Locate all the instances that matches the following description: bolt."}]}]
text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
images, videos = proc.process_vision_info(msgs)
inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt")

with torch.inference_mode():
    out = model.generate(
        pixel_values=inputs["pixel_values"].float(),
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws", None),
        tokenizer=tok, max_new_tokens=256, use_cache=True,
        generation_mode="hybrid", temperature=0.0, do_sample=False)
```

输出格式：`<ref>bolt</ref><box><x1><y1><x2><y2></box>...`，坐标归一化到 0–1000，乘图像宽高即得像素框（顺序为 **x1,y1,x2,y2**）。

## 4. 踩坑记录

### 4.1 transformers 5.24.0 与模型自带代码不兼容

模型使用 `trust_remote_code`，其 `modeling_locateanything.py` 依赖 `transformers.modeling_attn_mask_utils`，该模块在 **transformers 5.x 已被删除**，加载即报 `ModuleNotFoundError`。降到 README 指定的 **4.57.1** 后正常导入。

### 4.2 必须先 `import torch`，再 `import decord`

`decord` 会在 site-packages 里放一份 `libiomp5md.dll`，先导入 decord/cv2 再导入 torch 时，`torch\lib\c10.dll` 初始化失败：

```
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败 ... c10.dll
```

规避方式：所有入口脚本第一行 `import torch`。单独 `import torch` 或 torch→cv2→decord 顺序均正常。

### 4.3 默认注意力后端 `magi` 需要 CUDA

`config.json` 中 `_attn_implementation: magi`，CPU 下无法运行。必须显式设置 `cfg._attn_implementation = "sdpa"` 并传 `attn_implementation="sdpa"`。`la_flash` 同理。

### 4.4 wheel 集安装会降级既有环境

离线安装按 wheel 版本覆盖，导致 torch `2.15.0→2.12.0`、numpy `2.5.5→2.4.6`、opencv-python `5.1.0.95→4.13.0.92`、setuptools `84.2.1→81.0.0` 等。装完需回归一次原有脚本。

> 另：`opencv-python-headless` 与 `opencv-python` 同时存在会各提供一份 `cv2`，本项目按 README 要求装了 headless 4.11，实际生效的是 4.11。

### 4.5 内存与显存

- fp32 加载后工作集 17.0 GB，峰值 19.7 GB
- 动态量化后工作集升到 23.0 GB（Windows 堆未归还 OS，属正常现象），生成阶段峰值 24.9 GB
- 32 GB 内存余量很小，**并行跑其它视觉进程会有 OOM 风险**

### 4.6 远程执行的工程细节

- 通过 HTTP Shell 上传的 `.ps1` 若含中文路径，会以 UTF-8 无 BOM 落盘并被 PowerShell 按 ANSI 解析导致语法错误 → 脚本内避免中文常量，用 `Get-ChildItem` 按目录名搜索目标路径
- `(& $py xxx.py 2>&1 | Out-String) | Add-Content` 会**等进程结束才写日志**，无法观察进度 → 用 `python -u` 并直接重定向，或分阶段落到独立日志

## 5. 实测数据

| 指标 | fp32 | int8 动态量化 |
| --- | --- | --- |
| 权重加载 | 228.3 s | 241.2 s |
| 量化耗时 | — | 126.3 s |
| 单张推理（1022×672） | 71.97 s | 36.40 s / 36.65 s / 36.75 s |
| 加速比 | 1× | ≈1.96× |
| 进程工作集 | 17.0 GB（峰值 19.7 GB） | 23.0 GB（峰值 23.1 GB，生成后 24.9 GB） |
| 产物大小 | 7.26 GB（bf16 原件） | 4.79 GB（int8 state_dict，保存耗时 28 s） |
| 输出框数 | 40（日志截断，实际更多） | 43 |

| 输入（1022×672） | int8 输出 | fp32 输出 |
| --- | --- | --- |
| ![输入](locateanything-3b-images/input_1022x672.png) | ![int8](locateanything-3b-images/result_int8.png) | ![fp32](locateanything-3b-images/result_fp32.png) |

- int8：43 个框全部堆在画面左边缘，宽度 10–20 px 的细长竖条、彼此重叠，属于典型的退化输出
- fp32：多为 `<0><0><999><998>` 这类占满整幅的大框，同样退化
- 两者都在重复输出同一批框，说明模型对该工业近景 + 英文简短 prompt 完全失配，而非单纯量化损失

## 6. 复现步骤

```powershell
# 1) 离线 wheel（一次）
$py -m pip install --no-index --find-links <wheels目录> --upgrade <全部whl>

# 2) 补齐模型依赖
$py -m pip install transformers==4.57.1 accelerate safetensors sentencepiece einops tqdm regex protobuf peft decord lmdb

# 3) 运行（脚本须首行 import torch）
$py -u D:\JustSmartAuto\_diag\la_int8b.py

# 4) 结果图绘制
$py D:\JustSmartAuto\_diag\draw.py
```

相关脚本与日志保留在工控机 `D:\JustSmartAuto\_diag\`（`la_int8.py`、`la_int8b.py`、`la_fp32.py`、`draw.py`）与 `D:\JustSmartAuto\_ocl\`（结果图、框坐标 JSON、int8 state_dict）。

## 7. 建议

1. **不要在本项目上用该模型**：CPU 36 s/张 与精度双重不达标；即便有 GPU，场景适配仍是问题
2. 若后续仍要评估，优先做：原图直送（不缩放）、改用中文/具体部件描述的 prompt、用带标注的产线图做 few-shot 或微调
3. 需要更快的 CPU 推理时，可考虑 ONNX Runtime / OpenVINO 的静态 int8（需校准集），比 PyTorch 动态量化收益更大
4. `D:\JustSmartAuto\_ocl\locateanything3b_int8_dynamic.pt` 只是 state_dict，复用时仍需先构建 fp32 模型再 `quantize_dynamic` 后 `load_state_dict`
