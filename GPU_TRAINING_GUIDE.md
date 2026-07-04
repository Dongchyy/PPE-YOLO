# GPU 训练说明

本项目可以迁移到 Google Colab 或租赁 GPU 服务器上训练。推荐做法是：

- 代码：通过 GitHub 克隆。
- 数据集：通过 Google Drive、云盘、服务器硬盘或 `scp/rsync` 上传。
- 训练结果：训练完成后再同步回本地或云盘。

注意：大数据集和训练权重不要直接上传到 GitHub。

## 1. 需要准备哪些内容

GPU 环境中最终建议保持如下结构：

```text
YOLO/
├─ scripts/
├─ configs/
├─ requirements-gpu.txt
├─ images/                         # SH17 图片
├─ labels/                         # SH17 YOLO 标签
├─ train_files.txt
├─ val_files.txt
└─ data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours/
   ├─ data/                         # 691 个 mp4 视频
   ├─ samples.json
   ├─ frames.json
   ├─ metadata.json
   ├─ fiftyone.yml
   └─ README.md
```

迁移到 Colab/Linux 后，不要直接使用本地已有的 `configs/sh17_train.txt` 和 `configs/sh17_val.txt`，因为它们可能包含 Windows 路径。到 GPU 环境后需要重新执行：

```bash
python scripts/prepare_sh17.py
```

## 2. Colab 训练流程

### 2.1 挂载 Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 2.2 克隆 GitHub 仓库

```bash
cd /content
git clone https://github.com/Dongchyy/PPE-YOLO.git YOLO
cd /content/YOLO
```

### 2.3 安装依赖

```bash
pip install -r requirements-gpu.txt
```

### 2.4 从 Google Drive 复制数据集

下面的路径只是示例，你需要按自己 Drive 中的实际位置修改：

```bash
rsync -ah --info=progress2 /content/drive/MyDrive/YOLO_data/images/ ./images/
rsync -ah --info=progress2 /content/drive/MyDrive/YOLO_data/labels/ ./labels/
cp /content/drive/MyDrive/YOLO_data/train_files.txt ./train_files.txt
cp /content/drive/MyDrive/YOLO_data/val_files.txt ./val_files.txt

mkdir -p data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51
rsync -ah --info=progress2 \
  /content/drive/MyDrive/YOLO_data/Safe_and_Unsafe_Behaviours/ \
  ./data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours/
```

### 2.5 检查 GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果输出 `True` 并显示显卡名称，说明 GPU 可用。

### 2.6 重新生成 SH17 路径文件

```bash
python scripts/prepare_sh17.py
```

### 2.7 训练 PPE YOLO 模型

```bash
python scripts/train_ppe.py
```

默认结果会保存到：

```text
runs/detect/sh17_yolov8n/
```

其中最重要的文件是：

```text
runs/detect/sh17_yolov8n/weights/best.pt
```

### 2.8 训练行为识别模型

推荐先跑一个正式但不过大的版本：

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

如果显存不够，按下面顺序降低参数：

```text
batch-size: 8 -> 4 -> 2
clip-len: 16 -> 8
image-size: 112 先保持不变
```

训练结果会保存到：

```text
runs/behavior/r3d18_clip/
```

主要文件：

```text
best.pt
last.pt
r3d18_clip.pt
metrics.csv
classes.json
```

### 2.9 保存训练结果到 Google Drive

```bash
mkdir -p /content/drive/MyDrive/YOLO_runs
rsync -ah --info=progress2 ./runs/ /content/drive/MyDrive/YOLO_runs/
```

## 3. 租赁 GPU 服务器训练流程

### 3.1 在服务器上克隆代码

```bash
git clone https://github.com/Dongchyy/PPE-YOLO.git YOLO
cd YOLO
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

### 3.2 从本地上传数据集

在本地 Windows PowerShell 中执行，路径按自己的实际位置修改：

```powershell
scp -r D:\path\to\YOLO\images user@server:/home/user/YOLO/images
scp -r D:\path\to\YOLO\labels user@server:/home/user/YOLO/labels
scp D:\path\to\YOLO\train_files.txt user@server:/home/user/YOLO/train_files.txt
scp D:\path\to\YOLO\val_files.txt user@server:/home/user/YOLO/val_files.txt
scp -r D:\path\to\YOLO\data\raw\safe_unsafe_behaviours\huggingface\hub\Voxel51\Safe_and_Unsafe_Behaviours user@server:/home/user/YOLO/data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours
```

### 3.3 开始训练

```bash
python scripts/prepare_sh17.py
nvidia-smi
python scripts/train_ppe.py
python scripts/train_behavior_r3d18.py --train-per-class 9999 --test-per-class 9999 --clip-len 16 --image-size 112 --epochs 20 --batch-size 8 --lr 0.0003 --print-every 20
```

### 3.4 下载训练结果

在本地 Windows PowerShell 中执行：

```powershell
scp -r user@server:/home/user/YOLO/runs D:\path\to\YOLO\runs_gpu
```

## 4. 推荐训练参数

第一次正式 GPU 训练可以用：

```text
PPE YOLO:
  epochs: 100
  image size: 640
  batch: 16，显存不够则降低

行为识别 R3D-18:
  epochs: 20
  clip length: 16
  image size: 112
  batch: 8，如果是小显存 GPU 则用 4 或 2
```

最终展示或答辩前，可以考虑更长训练：

```text
PPE YOLO:
  epochs: 150-200

行为识别 R3D-18:
  epochs: 40-60
  learning rate: 0.0003
```

## 5. 常见问题

### YOLO 找不到图片

通常是因为 `configs/sh17_train.txt` / `configs/sh17_val.txt` 里还是旧路径。解决：

```bash
python scripts/prepare_sh17.py
```

### CUDA 不可用

先检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果 Colab 没有 GPU，需要在菜单中选择：

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

### 显存不足

优先降低行为识别训练的 `--batch-size`，再降低 `--clip-len`。PPE YOLO 训练则优先降低 `batch`。

