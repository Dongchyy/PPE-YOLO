# GPU 训练指南

本指南介绍 PPE 目标检测和视频行为分类的数据准备、训练及评估。所有命令均在项目根目录执行；两个任务分别运行，各自产生独立的模型权重。

## 1. 检查运行环境

在已配置 CUDA 版 PyTorch 的 GPU 环境中安装项目依赖：

```bash
python -m pip install -r requirements-gpu.txt
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`CUDA available: True` 表示当前 PyTorch 能使用 GPU。若为 `False`，先检查 GPU 是否分配成功以及 PyTorch 的 CUDA 支持，再启动训练。

## 2. 准备 SH17 数据

数据目录布局：

```text
PPE-YOLO/
|-- images/
|-- labels/
|-- train_files.txt
|-- val_files.txt
`-- configs/sh17.yaml
```

`images/` 保存图片，`labels/` 保存同名 YOLO 格式标注，两个列表文件记录训练集和验证集图片名称。

生成适用于当前机器的图片路径列表：

```bash
python scripts/prepare_sh17.py
```

检查终端中的 `missing images` 和 `missing labels`，确认数据完整后再训练。更换机器或移动项目目录后需要重新执行，因为生成的 `configs/sh17_train.txt` 和 `configs/sh17_val.txt` 使用绝对路径。

## 3. 训练 PPE 模型

使用项目默认训练配置：

```bash
python scripts/train_ppe.py
```

该脚本从 `yolov8n.pt` 开始训练，参数为 100 epochs、640 输入尺寸、batch 16、workers 0。首次使用时可能需要下载基础权重。

也可以直接使用 Ultralytics 命令指定 GPU 和训练参数。以下命令为 Linux GPU 服务器示例，与上面的脚本二选一：

```bash
yolo detect train model=yolov8n.pt data=configs/sh17.yaml epochs=100 imgsz=640 batch=32 device=0 workers=8 project="$PWD/runs/sh17" name=yolov8n_640_e100_b32
```

显存不足时降低 `batch`。训练时长与硬件、数据读取速度和参数有关，应根据前几轮包含验证的实际耗时估算。

默认脚本的首次训练结果位于 `runs/detect/sh17_yolov8n/`；上面的自定义命令使用 `runs/sh17/yolov8n_640_e100_b32/`。重复运行可能自动生成带编号的目录，以日志中的路径为准。

主要结果：

| 文件 | 用途 |
| --- | --- |
| `weights/best.pt` | 根据验证指标选出的模型，用于效果评估和预测 |
| `weights/last.pt` | 最后一轮模型 |
| `results.csv` | 各轮训练损失和验证指标 |
| `results.png` | 训练曲线 |

使用默认脚本训练得到的模型预测图片：

```bash
python scripts/predict_ppe.py --weights runs/detect/sh17_yolov8n/weights/best.pt --source test1.jpg --conf 0.25 --name ppe_eval
```

评估时同时关注 Precision、Recall、mAP50 和 mAP50-95，以及护目镜、手套、口罩、防护服等目标类别的单独指标。整体平均指标不能代表每一种防护装备的识别效果。

## 4. 准备行为视频数据

训练脚本使用以下目录：

```text
data/raw/safe_unsafe_behaviours/huggingface/hub/Voxel51/Safe_and_Unsafe_Behaviours/
|-- data/
|-- samples.json
|-- frames.json
|-- metadata.json
`-- fiftyone.yml
```

脚本直接读取 `samples.json` 中的 `samples`，通过 `filepath` 定位视频，通过 `ground_truth.label` 获取类别，通过 `tags` 中的 `train` 和 `test` 区分数据划分。

当前实现把数据集的 `test` 划分用于每轮验证和最优模型选择，因此日志中的 `val_acc` 不是独立测试集成绩。需要报告泛化能力时，应另设未参与调参和模型选择的测试数据。

## 5. 训练行为模型

以下命令使用每类最多 9999 个样本，对当前数据集相当于取对应划分的全部视频：

```bash
python scripts/train_behavior_r3d18.py --train-per-class 9999 --test-per-class 9999 --clip-len 16 --image-size 112 --epochs 20 --batch-size 8 --lr 0.0003 --print-every 20
```

| 参数 | 含义 |
| --- | --- |
| `--train-per-class` / `--test-per-class` | 每类参与训练、验证的视频数量上限 |
| `--clip-len` | 每次从视频抽取的帧数 |
| `--image-size` | 每帧输入尺寸 |
| `--epochs` | 训练轮数 |
| `--batch-size` | 每批视频片段数量 |
| `--lr` | AdamW 学习率 |
| `--print-every` | 训练日志输出间隔，单位为 step |

模型使用随机初始化的 R3D-18，没有加载动作识别预训练权重。训练时随机抽取连续帧，验证和预测时在视频中间约 80% 范围内均匀采样；两者时间跨度不同，分析效果时需要考虑这一差异。

结果固定保存在 `runs/behavior/r3d18_clip/`：

| 文件 | 用途 |
| --- | --- |
| `best.pt` | 已完成验证的轮次中，验证准确率最高的模型；相同分数时更新 |
| `last.pt` | 最近一轮完整训练和验证后的模型 |
| `r3d18_clip.pt` | 全部训练正常结束时保存的最终模型，不保证是最优模型 |
| `metrics.csv` | 每轮训练和验证的 loss、accuracy |
| `classes.json` | 类别名称与编号映射 |

再次启动脚本会覆盖该固定目录中的同名结果。当前脚本不支持断点续训；中断时未完成的轮次不会保存，已写入的 `best.pt` 和 `last.pt` 可以用于预测。

预测单个视频：

```bash
python scripts/predict_behavior.py --weights runs/behavior/r3d18_clip/best.pt --source path/to/video.mp4 --topk 3
```

## 6. 观察训练与验证效果

在另一个服务器终端持续查看 GPU 状态：

```bash
watch -n 1 nvidia-smi
```

视频脚本使用 OpenCV 解码，且 DataLoader 的 `num_workers=0`。若 GPU 利用率长期偏低而 CPU 忙碌，需要检查视频读取和解码耗时，单纯增加训练轮数或 batch 不一定能解决速度问题。

比较多轮的训练、验证损失与准确率，不依据单个 batch 判断模型效果。若训练指标改善而验证指标持续恶化，应优先检查数据划分、采样方式和学习率，并用 `best.pt` 进行独立样本验证。

## 7. 实验室场景验证

使用未参与训练和模型选择的实验室图片、视频进行验证，覆盖不同人员、视角、光照、遮挡及装备组合。PPE 检测重点检查漏检、误检和多人场景；行为分类重点检查类别是否适用于实际场景及错误分类样例。

若需要使用自采数据微调，应按人员、拍摄场景或原始视频划分训练、验证和测试数据，避免把同一视频的相邻帧分到不同集合。两个模型当前独立输出结果，尚未实现联合报警判断。
