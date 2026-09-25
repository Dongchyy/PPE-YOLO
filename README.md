# PPE-YOLO

面向实验室安全场景的计算机视觉原型，结合 **YOLOv8 防护装备检测**与 **R3D-18 视频行为分类**，用于观察人员防护装备、识别工作场所行为，并提供基础安全复核提示。

## 项目功能

| 模块 | 输入 | 实现内容 | 输出 |
| --- | --- | --- | --- |
| 防护装备检测 | 图片、视频或图片文件夹 | 使用 SH17 训练的 YOLOv8n 定位人员及防护装备 | 检测框、类别、置信度、目标数量及可视化结果 |
| PPE 规则提示 | 图片或视频帧的检测结果 | 检查画面中是否检测到人员、眼镜、手套、口罩及防护服 | 缺失目标提示，供人工复核 |
| 视频行为分类 | 单个视频文件 | 抽取视频帧，使用 3D ResNet-18 进行片段分类 | Top-K 行为类别、预测分数及 JSON 格式结果 |
| 模型训练 | 标注图片或带类别标签的视频 | 数据准备、训练、验证及模型保存 | 模型权重、训练指标和类别映射 |

## 防护装备检测

基于 SH17 的 17 类目标训练 YOLOv8n，预测脚本默认关注以下目标：

| 检测目标 | 模型类别 |
| --- | --- |
| 人员 | `person` |
| 眼镜 / 护目镜相关目标 | `glasses` |
| 手套 | `gloves` |
| 口罩 | `face-mask` |
| 实验服 / 防护服相关目标 | `medical-suit`、`safety-suit` |

支持调整置信度阈值、自定义检测类别，也可以显示全部 17 类目标。结果保存为带标注的图片或视频，终端输出各类目标数量及规则提示。

模型沿用 SH17 的类别定义：`glasses` 不能证明眼镜具有防护能力，`medical-suit` 和 `safety-suit` 也不等同于所有实验室白大褂。具体实验室场景的识别效果需要使用自采数据验证。

## 视频行为分类

使用 Safe and Unsafe Behaviours 数据集训练 R3D-18，通过三维卷积处理视频片段的空间与时间信息，学习数据集中安全与不安全行为的类别差异。类别名称从数据集标签读取，预测时使用权重中的类别映射。

当前训练实现采用随机初始化的 R3D-18、视频帧采样与归一化、随机水平翻转、交叉熵损失和 AdamW 优化器，并记录每轮训练与验证的损失及准确率。

行为模型与 PPE 模型分别训练、分别预测，权重相互独立。视频分类输出的是片段级类别，不包含行为发生位置、人员身份或操作权限判断。

## 快速体验

在项目根目录安装依赖：

```bash
python -m pip install -r requirements-gpu.txt
```

使用训练好的 PPE 模型检测示例图片：

```bash
python scripts/predict_ppe.py --weights exported_models/sh17_yolo_best.pt --source test1.jpg --conf 0.25 --name test1_ppe
```

首次使用该输出名称时，可视化结果位于 `runs/predict/test1_ppe/`。重复运行时，Ultralytics 可能自动给目录名添加编号，以实际推理日志为准。将 `--source` 改为其他图片、视频或图片文件夹即可更换输入。

常用预测选项：

| 参数 | 用途 |
| --- | --- |
| `--conf 0.25` | 设置检测置信度阈值 |
| `--target-classes person,glasses,gloves` | 仅检测指定类别 |
| `--all-classes` | 显示 SH17 全部类别 |
| `--no-rules` | 仅查看检测结果，关闭规则提示 |
| `--name test1_ppe` | 指定结果目录名称 |

只查看部分目标时，建议同时使用 `--no-rules`，避免将未参与检测的类别计入缺失提示。

完成行为模型训练后，对视频进行分类：

```bash
python scripts/predict_behavior.py --weights runs/behavior/r3d18_clip/best.pt --source path/to/video.mp4 --topk 3
```

PPE 推理可在 CPU 环境运行；行为预测脚本自动选择可用的 CUDA GPU，否则使用 CPU。数据准备、完整训练命令和指标说明见 [GPU 训练指南](GPU_TRAINING_GUIDE.md)。

## 实现范围

- PPE 规则基于整张画面的目标数量，尚未实现装备与人员的逐人关联。多人场景中检测到一副手套，不代表每个人都戴了手套。
- “未检测到”可能来自遮挡、目标过小或模型漏检，不能直接等同于“未佩戴”；检测到装备也不能证明佩戴正确。
- 行为模型学习的是已有数据集中的类别，不能据此识别任意实验室异常。迁移到实验室场景需要独立评估，必要时补充标注与微调。
- 当前已实现命令行训练与预测，尚未实现两个模型的自动融合、ESP32 传感器接入或实时报警系统。

## 技术组成

- **目标检测**：Ultralytics YOLOv8n。
- **视频分类**：PyTorch、TorchVision R3D-18。
- **视频处理**：OpenCV。
- **数据处理**：NumPy、YAML 配置与 JSON 标签。

## 项目结构

```text
PPE-YOLO/
|-- configs/sh17.yaml                  # SH17 类别与数据路径配置
|-- scripts/
|   |-- prepare_sh17.py                # 生成训练、验证图片路径列表
|   |-- train_ppe.py                   # YOLOv8n 训练
|   |-- predict_ppe.py                 # PPE 检测与规则提示
|   |-- train_behavior_r3d18.py        # R3D-18 视频分类训练
|   |-- predict_behavior.py            # 视频片段分类预测
|   `-- download_safe_unsafe_behaviours.py
|-- exported_models/sh17_yolo_best.pt   # PPE 推理模型
|-- test1.jpg / test2.jpg / test3.jpg    # 示例图片
|-- requirements.txt                   # PPE 基础依赖
|-- requirements-gpu.txt               # 两个模块的运行依赖
`-- GPU_TRAINING_GUIDE.md               # 数据准备与训练指南
```

## 数据来源

- [SH17 Dataset](https://github.com/ahmadmughees/SH17dataset)：人体与个人防护装备目标检测数据集。
- [Safe and Unsafe Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours)：工作场所安全与不安全行为视频数据集。

这两个公开数据集主要来自工业和工作场所场景，本项目将其用于实验室安全方向的原型探索。
