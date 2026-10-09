# nanoGPT 任务提交

> 提交人：Henry-he-08　　提交日期：2026-10-09

---

## 任务要求对照表

### 第一部分 · Transformer 概念理解

> 📝 **手写笔记**：[`04_Transformer学习笔记（手写）.pdf`](<04_Transformer学习笔记（手写）.pdf>) **第 2 页**
> ——Tensor 记号（B/`batch_size`、`d-model`、`h`、`d_k=d_v`）、整体数据流、Q/K/V 含义、
> Attention 六步（含 `reshape/transpose` 拆多头）、Add & Norm、FFN、Decoder 与 Encoder 的区别

| 任务要求 | 在哪看 |
|---|---|
| Transformer Encoder–Decoder 架构 | [`02_论文与源码理解.md`](02_论文与源码理解.md) §1.1 |
| Encoder Block / Decoder Block 的主要组成部分 | 同上 §1.1 |
| 说明 Q / K / V 的含义及 Attention 基本计算过程 | 同上 §1.2（含"为什么除以 √d_k"） |
| 【进阶】为什么 Decoder 需要 Causal / Masked Attention | 同上 §1.7 |
| 【进阶】Multi-Head Attention 与单头 Attention 的区别 | 同上 §1.3 |
| 【进阶】Attention / MLP / Residual / LayerNorm 各自的功能 | 同上 §3.7 |

### 第二部分 · nanoGPT 项目与模型结构

> 📝 **手写笔记第 3 页**：GPT 内部结构树（Token/Position Embedding → Decoder Block ×N
> → LayerNorm → LM Head → 下一个 token 概率），以及 Decoder Block 内部逐层展开
> 📝 **手写笔记第 5 页**：代码要点（`_init_weights` 的 `1/√(2·n_layer)`、`hasattr`、
> 优化器参数分组）

| 任务要求 | 在哪看 |
|---|---|
| 1. 项目结构：`model.py` / `train.py` / `sample.py` / `configurator.py` / `data/` / `config/` 各负责什么 | [`02_论文与源码理解.md`](02_论文与源码理解.md) §2.1 |
| 2. nanoGPT 模型结构：按 `model.py` 整理树状结构 | 同上 §2.3（结构树） |
| 2. 说明各模块的作用 | 同上 §2.3（逐模块）+ §3.7（速查表） |
| 环境部署并运行 nanoGPT | [`01_实验报告.md`](01_实验报告.md) §1 + 本文「环境」节 |

### 第三部分 · 【进阶】Tensor Shape

> 📝 **手写笔记第 4 页**：带真实数字的完整形状追踪
> （`(B,L)` → `(B,L,768)` → reshape `(B,L,12,64)` → transpose `(B,12,L,64)`
> → scores `(B,12,L,L)` → `out (B,L,768)` → FFN `(B,L,3072)` → logits `(B,L,50257)`）

| 任务要求 | 在哪看 |
|---|---|
| 3. 追踪 8 个关键 tensor 的 shape | [`02_论文与源码理解.md`](02_论文与源码理解.md) **§3.2（按题目清单速查）** + §3.3（25 步推导） |
| 3. **解释每一次 `view` / `transpose` / `split` 为什么成立** | 同上 §3.5（四个操作的原理） |
| 3. 一个 token 的完整旅程 | 同上 §3.6 |

### 提交的实测结果

| 任务要求 | 在哪看 |
|---|---|
| 训练结果（loss 日志） | [`01_实验报告.md`](01_实验报告.md) §3 + [`outputs/03_loss_table.txt`](outputs/03_loss_table.txt) |
| 文本生成结果 | 同上 §4 + [`outputs/01_generated_trained.txt`](outputs/01_generated_trained.txt) |
| 进阶 · 生成文本 vs 未训练模型 | 同上 §5 + [`outputs/02_generated_untrained.txt`](outputs/02_generated_untrained.txt) |
| 进阶 · 训练前后 loss 变化是否符合预期 | 同上 §3.3 |
| 进阶 · train loss 与 val loss 分别反映什么 | 同上 §3.4 |
| 进阶 · temperature / top_k 的作用 | 同上 §4.4 |
| 进阶 · n_layer / n_head / n_embd / block_size 的含义 | 同上 §6 |
| 体现自己的理解过程 | [`03_学习过程记录.md`](03_学习过程记录.md) |

---

## 一句话结论

在 Windows + RTX 5060 上完整跑通了 nanoGPT 的
**数据准备 → 训练 5000 步 → 生成文本** 全流程。

模型 10.65M 参数，在莎士比亚数据集上训练 10 分钟，
loss 从 **4.29 降到 0.62**（train），能生成格式正确、
角色名真实的莎剧风格文本。

---

## 文件结构

```
nanogpt_submit/
├── README.md                  ← 你在这里（导航 + 任务对照）
├── 01_实验报告.md              主体：环境 / 数据 / 训练 / 生成 / 进阶分析
├── 02_论文与源码理解.md         论文要点 / 模块职责 / 25 步形状追踪表
├── 03_学习过程记录.md           过程记录：卡点、走过的弯路、自评
├── 04_Transformer学习笔记（手写）.pdf   ★ 手写笔记：Tensor 记号 / 数据流 /
│                                      GPT 结构树 / 形状追踪（6 页）
│
├── logs/
│   └── 原始训练日志.txt         完整 5000 步原始输出（未加工）
│
├── outputs/
│   ├── 01_generated_trained.txt    训练后模型，3 个 temperature 的生成结果
│   ├── 02_generated_untrained.txt  未训练模型对照（同一 prompt）
│   └── 03_loss_table.txt           loss 变化表（带最优步标注）
│
└── code/
    ├── model.py                    模型定义（GPT / Block / Attention / MLP / LayerNorm）
    ├── train.py                    训练脚本
    ├── sample.py                   官方生成脚本
    ├── configurator.py             配置加载器
    ├── demo_generate.py            ★ 精简版生成脚本（参数已填好，可直接运行）
    ├── compare_untrained.py        未训练对照实验
    ├── config/
    │   └── train_shakespeare_char.py   本次训练用的配置
    └── data/
        └── shakespeare_char/
            ├── input.txt               莎士比亚语料（1.1 MB，已随包提供）
            └── prepare.py              数据准备脚本（含本次打的补丁）
```

**本包已完全自包含**：数据、代码、配置、脚本都在里面，**不需要联网**，
解压后按下面三步即可跑出与本次实验一致的结果。

**注**：训练好的 checkpoint（`ckpt.pt`，123 MB）未附在包内，
见下方「关于 checkpoint 文件」一节。

---

## 环境（实测）

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 |
| Python | 3.12.14 (CPython) |
| PyTorch | 2.14.0+cu130 |
| CUDA | 13.0，可用 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 算力 | sm_120 (compute capability 12.0) |
| 显存 | 7.96 GB |
| CPU | 24 逻辑核心 |

---

## 复现步骤（三条命令，约 10 分钟）

```bash
# 0. 把本包的 code/ 全部拷到一个工作目录，并 cd 进去
#    （code/ 就是一份可运行的 nanoGPT 精简副本）
mkdir D:\nanogpt_work
xcopy /E /I code D:\nanogpt_work
cd /d D:\nanogpt_work

# 1. 准备数据（只需一次，约 5 秒）
python data/shakespeare_char/prepare.py

# 2. 训练 5000 步（约 10 分钟）
python train.py config/train_shakespeare_char.py --compile=False

# 3. 生成文本（几秒）
python demo_generate.py
```

> **本机已验证路径**：本次实验是在 `D:\ML\nanogpt_orig` 下跑通的，
> 上面三条命令在该目录下同样可用。

### 只想看生成结果，不想重训？

**在包根目录**（也就是你解压出来的 `nanogpt_submit/`）直接跑就行，
脚本会**自动搜索**已有的 checkpoint：

```bash
python code/demo_generate.py        # 生成文本
python code/compare_untrained.py    # 未训练 vs 训练后 对照
```

搜索不到时，它会打印确切的解决办法；也可以手动指定：

```bash
python code/demo_generate.py --ckpt D:\ML\nanogpt_orig\out-shakespeare-char\ckpt.pt
```

---

## 关于 checkpoint 文件

训练产出的 `out-shakespeare-char/ckpt.pt` 约 **123 MB**，**未附在包内**，
原因有二：

1. 体积大，多数提交平台（含 GitHub 的 100 MB 单文件上限）会拒收；
2. 按上面的复现步骤跑 10 分钟即可重新生成，结果完全一致。

**如果评选方要求提供训练好的权重**，把 `out-shakespeare-char/` 整个目录
单独打包或上传网盘即可，`demo_generate.py` 会自动读取它。

### 没放 checkpoint 时，脚本怎么找到它？

**`demo_generate.py` 和 `sample.py` 都会自动搜索**，你不必手动传路径。搜索顺序：

```
① 命令行 --ckpt 参数（最高优先）
② 环境变量 NANOGPT_CKPT
③ 以下候选位置（本文件目录 / 当前目录 / 向上 2 层 + 这些层的同级子目录）
      <某个根>/out-shakespeare-char/ckpt.pt
      <某个根>/out-*/ckpt.pt          ← 训练目录名不同也能找到
```

**举例**：本文件在 `D:\ML\nanogpt_submit\code\` 时，
向上找到 `D:\ML`，再扫它的同级子目录 → 命中
`D:\ML\nanogpt_orig\out-shakespeare-char\ckpt.pt`。

**如果搜索完还是没有**，脚本不会抛晦涩的 `FileNotFoundError`，
而是打印**确切的两条命令**（直接指定路径 / 先训练一次）。

**手动指定路径**（最快）：

```bash
python demo_generate.py --ckpt D:\ML\nanogpt_orig\out-shakespeare-char\ckpt.pt
```

### 词表（meta.pkl）也做了兜底

字符级数据集的 `meta.pkl` 内容就是 `sorted(set(input.txt))` 得到的 65 个字符，
**完全可以现算**（已实测：现算结果与 `meta.pkl` 逐项相同）。

所以本包的脚本在 `meta.pkl` 不存在时，**会自动从 `input.txt` 现算**，
不必先跑 `prepare.py`，也避免错误地退回 GPT-2 词表（50304 ≠ 65）。

---

## ⚠️ 四个必须知道的坑（否则一定失败）

### 1. Windows 上必须加 `--compile=False`

原版 `train.py` 默认 `compile = True`（用 `torch.compile` 加速），
但 **Windows 上没有 Triton**，会直接报错。

### 2. 数据必须用官方 `prepare.py` 生成

要求 **uint16 二进制** + **`meta.pkl`**（词表）。

- 存成 int64 会报
  `ValueError: Size of available data is not a multiple of the data-type size.`
- 缺 `meta.pkl` 会让模型**默认使用 GPT-2 的词表 50304**：
  ```
  defaulting to vocab_size of GPT-2 to 50304   ← 错
  found vocab_size = 65 (inside ...meta.pkl)    ← 对
  ```
  参数量会从 10.65M 变成 29.94M，**差 3 倍**。

### 3. `prepare.py` 依赖 `requests`、`sample.py` 依赖 `tiktoken`

两者都只在特定分支用得上（下载数据 / 加载 GPT-2 权重），
本环境已改成**可选导入**，避免 `ModuleNotFoundError`。

### 4. `[错误] 找不到 out-shakespeare-char/ckpt.pt`

**这不是代码坏了，是 checkpoint 不在当前目录。**

checkpoint 只在**训练产出它的那个目录**里，而且本包**故意不带**
（123 MB，见上方「关于 checkpoint 文件」）。所以：

- 原来在 `nanogpt_orig` 里跑得通，换到别的目录就跑不通 —— **正常现象**
- `demo_generate.py` / `sample.py` 现在会**自动搜索**多个候选位置，
  并打印确切的解决办法，不会再抛晦涩的 `FileNotFoundError`

**手动指定路径**（最快）：

```bash
python demo_generate.py --ckpt D:\ML\nanogpt_orig\out-shakespeare-char\ckpt.pt
```

**或先训练一次**（约 10 分钟）：

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --compile=False
```

---

## 本次踩到的其它问题

- GitHub raw 直连不稳定 → 改用 `cdn.jsdelivr.net/gh/...` 镜像
- 参数名 `model_args` 与 `config` 容易混（一个是模型结构、一个是训练流程）
- checkpoint 的键名是 `model` 不是 `state_dict`
- `.vscode/launch.json` 里的 `${workspaceFolder}` 要求用对工作区目录
