# PPE-YOLO

YOLO-based lab safety prototype for PPE detection and workplace/lab behavior recognition.

## Main Parts

- PPE detection with SH17-style YOLO labels
- Behavior recognition with Safe and Unsafe Behaviours video clips
- CPU smoke-test scripts and GPU-ready training scripts
- Colab/rented-GPU deployment guide

## Project Structure

```text
configs/
  sh17.yaml
scripts/
  prepare_sh17.py
  train_ppe.py
  predict_ppe.py
  train_behavior_r3d18.py
  download_safe_unsafe_behaviours.py
requirements.txt
requirements-gpu.txt
GPU_TRAINING_GUIDE.md
```

Large datasets, training runs, caches, and model weights are intentionally excluded from Git.

## Prepare SH17

After copying the SH17 dataset into the project root, regenerate path files:

```bash
python scripts/prepare_sh17.py
```

Then train:

```bash
python scripts/train_ppe.py
```

## Train Behavior Model

Place the Safe and Unsafe Behaviours dataset at:

```text
data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours/
```

Run a GPU training job:

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

See [GPU_TRAINING_GUIDE.md](GPU_TRAINING_GUIDE.md) for Colab and rented GPU instructions.

