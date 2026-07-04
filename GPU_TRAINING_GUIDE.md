# GPU Training Guide

This project can be moved to Colab or a rented GPU server. Use Git/GitHub for code and Google Drive, cloud disk, or `scp/rsync` for datasets and checkpoints.

## 1. What To Upload

Upload the code and these datasets:

```text
YOLO/
├─ scripts/
├─ configs/
├─ requirements-gpu.txt
├─ yolov8n.pt
├─ images/                         # SH17 images
├─ labels/                         # SH17 YOLO labels
├─ train_files.txt
├─ val_files.txt
└─ data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours/
   ├─ data/                         # 691 mp4 videos
   ├─ samples.json
   ├─ frames.json
   ├─ metadata.json
   ├─ fiftyone.yml
   └─ README.md
```

Do not rely on the existing `configs/sh17_train.txt` and `configs/sh17_val.txt` after moving to Linux/Colab, because they may contain Windows paths. Regenerate them on the GPU machine.

## 2. Colab Workflow

### Option A: Code From GitHub, Data From Google Drive

Mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone or upload this project:

```bash
cd /content
git clone <your-repo-url> YOLO
cd /content/YOLO
```

Install dependencies:

```bash
pip install -r requirements-gpu.txt
```

Copy datasets from Drive into the expected project paths. Adjust the source paths to your Drive layout:

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

Regenerate SH17 config paths:

```bash
python scripts/prepare_sh17.py
```

Check GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Train SH17 PPE detector:

```bash
python scripts/train_ppe.py
```

Train video behavior model with a larger GPU setting:

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

If GPU memory is insufficient, reduce in this order:

```text
batch-size: 8 -> 4 -> 2
clip-len: 16 -> 8
image-size: 112 stays unchanged first
```

Save results back to Drive:

```bash
mkdir -p /content/drive/MyDrive/YOLO_runs
rsync -ah --info=progress2 ./runs/ /content/drive/MyDrive/YOLO_runs/
```

## 3. Rented GPU Server Workflow

On the server:

```bash
git clone <your-repo-url> YOLO
cd YOLO
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

Upload datasets from your computer. Example from local PowerShell:

```powershell
scp -r D:\path\to\YOLO\images user@server:/home/user/YOLO/images
scp -r D:\path\to\YOLO\labels user@server:/home/user/YOLO/labels
scp D:\path\to\YOLO\train_files.txt user@server:/home/user/YOLO/train_files.txt
scp D:\path\to\YOLO\val_files.txt user@server:/home/user/YOLO/val_files.txt
scp -r D:\path\to\YOLO\data\raw\safe_unsafe_behaviours\huggingface\hub\Voxel51\Safe_and_Unsafe_Behaviours user@server:/home/user/YOLO/data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours
```

Then run:

```bash
python scripts/prepare_sh17.py
nvidia-smi
python scripts/train_ppe.py
python scripts/train_behavior_r3d18.py --train-per-class 9999 --test-per-class 9999 --clip-len 16 --image-size 112 --epochs 20 --batch-size 8 --lr 0.0003 --print-every 20
```

Download checkpoints after training:

```powershell
scp -r user@server:/home/user/YOLO/runs D:\path\to\YOLO\runs_gpu
```

## 4. Expected Outputs

PPE YOLO results:

```text
runs/detect/sh17_yolov8n/
├─ weights/best.pt
├─ weights/last.pt
└─ results.csv
```

Behavior video results:

```text
runs/behavior/r3d18_clip/
├─ best.pt
├─ last.pt
├─ r3d18_clip.pt
├─ metrics.csv
└─ classes.json
```

## 5. Recommended GPU Settings

For a first real run:

```text
PPE YOLO:
  epochs: 100
  image size: 640
  batch: 16 or auto-adjust by GPU memory

Behavior R3D-18:
  epochs: 20
  clip length: 16
  image size: 112
  batch: 8 on 16GB GPU, 4 on smaller GPU
```

For a longer final run:

```text
PPE YOLO:
  epochs: 150-200

Behavior R3D-18:
  epochs: 40-60
  learning rate: 0.0003
```
