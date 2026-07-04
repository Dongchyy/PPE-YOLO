# PPE-YOLO

这是一个面向实验室安全场景的原型项目，包含两条主线：

- PPE 防护装备检测：基于 SH17 数据集训练 YOLO 模型，识别人员、护目镜、口罩、手套、防护服、安全帽等目标。
- 异常/安全行为识别：基于 Safe and Unsafe Behaviours 视频数据集训练视频片段分类模型，用于识别工作场所安全/不安全行为。

本仓库只保存代码、配置和说明文档。大数据集、训练输出、缓存文件和模型权重不会上传到 GitHub。

## 项目结构

```text
configs/
  sh17.yaml                         # SH17 YOLO 数据集配置
scripts/
  prepare_sh17.py                   # 生成 SH17 训练/验证图片路径
  train_ppe.py                      # 训练 PPE YOLO 模型
  predict_ppe.py                    # 使用 PPE 模型预测图片/视频
  train_behavior_r3d18.py           # 训练 3D ResNet-18 视频行为识别模型
  download_safe_unsafe_behaviours.py # 下载 Hugging Face 行为视频数据集
requirements.txt                    # 本地基础依赖
requirements-gpu.txt                # GPU/Colab 训练依赖
GPU_TRAINING_GUIDE.md               # Colab 和租赁 GPU 训练说明
```

## 不上传的内容

以下内容已经通过 `.gitignore` 排除：

```text
images/
labels/
meta-data/
voc_labels/
data/
runs/
weights/
*.pt
*.pth
*.onnx
yolov8n.pt
labels.cache
```

所以 GitHub 仓库不会包含 SH17 图片、视频数据集、训练结果或权重文件。

## 准备 SH17 数据集

将 SH17 数据放在项目根目录下，保持类似结构：

```text
YOLO/
├─ images/
├─ labels/
├─ train_files.txt
└─ val_files.txt
```

然后生成 YOLO 训练所需的路径文件：

```bash
python scripts/prepare_sh17.py
```

开始训练 PPE 检测模型：

```bash
python scripts/train_ppe.py
```

训练结果会输出到：

```text
runs/detect/sh17_yolov8n/
```

## 准备行为识别数据集

Safe and Unsafe Behaviours 数据集需要放在：

```text
data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours/
```

目录中应包含：

```text
data/              # 691 个 mp4 视频
samples.json       # 视频路径、标签、train/test 划分
frames.json
metadata.json
fiftyone.yml
README.md
```

运行视频行为识别训练：

```bash
python scripts/train_behavior_r3d18.py \
  --train-per-class 9999 \
  --test-per-class 9999 \
  --clip-len 16 \
  --image-size 112 \
  --epochs 20 \
  --batch-size 8 \
  --lr 0.0003 \
  --print-every 20
```

如果显存不足，优先降低 `--batch-size`，例如从 `8` 改成 `4` 或 `2`。

训练结果会输出到：

```text
runs/behavior/r3d18_clip/
```

## 在 GPU 上训练

如果本机没有 GPU，推荐把代码上传到 GitHub，把数据集放到 Google Drive、云盘或租赁 GPU 服务器硬盘，然后在 GPU 环境运行训练。

详细步骤见：

[GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md)
