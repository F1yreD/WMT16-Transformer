# WMT16-Transformer

基于 PyTorch 的 Transformer（Encoder-Decoder）实现与实验工程，面向 **德语→英语** 机器翻译任务。项目包含从数据预处理（Moses 分词 + BPE）、训练、验证/测试、TensorBoard 可视化到 BLEU 评估（BLEU-1 / BLEU-4）的完整流程，主要以 Jupyter Notebook 形式组织。

数据集来源：WMT16 Multimodal Task（DE-EN）  
https://www.statmt.org/wmt16/multimodal-task.html#task1

- testing cross entropy loss: 3.419556591629982
- BLEU-1 (sentence avg): 0.5672879182442075
- BLEU-4 (sentence avg): 0.21087645611642775
- BLEU-4 (corpus): 0.2244893350820118

## 主要内容

- Transformer 模型实现（含注意力可视化相关输出）
- 数据处理：MosesTokenizer 分词、subword-nmt 的 joint BPE 与应用
- 训练/验证：带 Padding Mask 的 CrossEntropy（支持 label smoothing）
- 评估：测试集 loss + BLEU-1 / BLEU-4（含 smoothing，避免高阶 n-gram 缺失导致 BLEU=0）
- 监控：TensorBoard 记录 loss / val_loss / lr 等

## 仓库结构

- `transformer-enhance.ipynb`：主 notebook（推荐使用这个）
- `transformer_带bleu-aliyun.ipynb`：带 BLEU 的另一个版本（逻辑相近，用于对照）
- `data_multi30k.py`：对原始平行语料进行 Moses 分词并输出到目标目录
- `data_multi30k.sh`：基于 `subword-nmt` 进行 joint BPE 学习与应用（更适合 Linux/Colab 环境）
- `wmt16/`：数据目录（建议放置预处理后的 `.bpe` 与 `vocab/bpe.*` 文件）
- `checkpoints/`：训练过程保存的权重（默认被 `.gitignore` 忽略）
- `runs/`：TensorBoard 日志（默认被 `.gitignore` 忽略）

## 环境依赖

建议使用 Conda/Miniconda 创建独立环境（Python 3.10+ 通常可用；Notebook 内也会打印关键库版本）。

核心依赖（按 notebook 实际使用）：

- PyTorch
- numpy / pandas / matplotlib / tqdm
- sacremoses（MosesTokenizer）
- subword-nmt（BPE 学习与应用）
- nltk（BLEU 计算）
- tensorboard（可选，用于可视化）

安装示例（按需选择）：

```bash
pip install sacremoses subword-nmt nltk tensorboard tqdm numpy pandas matplotlib
```

## 数据准备（DE-EN）

### 1) Moses 分词

将原始数据目录作为 `--pair_dir`，输出目录作为 `--dest_dir`（会生成 `train/val/test` 的分词结果）：

```bash
python data_multi30k.py --pair_dir <raw_pair_dir> --dest_dir <dest_dir> --src_lang de --trg_lang en
```

### 2) BPE（joint BPE）

项目中提供了 `data_multi30k.sh` 作为参考流程（Linux/Colab 更方便）。其逻辑包括：

- 合并训练集 src/trg 到一个文件
- `subword-nmt learn-joint-bpe-and-vocab` 学习 joint BPE 与词表
- `subword-nmt apply-bpe` 分别对 train/val/test 的 src/trg 应用 BPE

Windows 下如果无法直接运行 `.sh`，可以参考脚本内容手动执行其中的 `subword-nmt` 命令，或在 WSL/Git Bash 环境运行。

最终你需要在 `wmt16/` 下具备类似文件：

- `train_src.bpe / train_trg.bpe`
- `val_src.bpe / val_trg.bpe`
- `test_src.bpe / test_trg.bpe`
- `vocab`、`bpe.20000`（或对应的 BPE codes 文件）

## 训练与评估

### Notebook 使用方式

推荐打开 `transformer-enhance.ipynb` 并从上到下依次运行（Notebook 对执行顺序敏感，尤其会影响随机种子与模型初始化）。

训练阶段要点：

- `loss_fct(logits, decoder_labels, padding_mask=decoder_labels_mask)`：训练/验证 loss 会使用 `decoder_labels_mask` 屏蔽 PAD 位置，避免 padding 影响 loss。
- 记录 loss 时使用 `loss.cpu().item()`，避免把计算图/显存错误地留在列表中。

评估阶段要点：

- BLEU 需要对 BPE 的 `@@` 做还原（把被 BPE 拆开的子词合并回单词），否则 BLEU 会显著偏低。
- 当前 notebook 中同时计算并输出：
  - BLEU-1（sentence avg）
  - BLEU-4（sentence avg）
  - BLEU-4（corpus）

## 常见问题

### 1) 两个 notebook 的训练 loss 为什么会不一样？

即使在同一环境，Notebook 中“是否重复运行过某些 cell（尤其是会消耗随机数的 cell）”也会导致模型初始化不同，从而使训练曲线出现差异。建议：

- Restart Kernel 后按顺序运行一遍；或
- 在“创建模型之前”再次设置随机种子，确保初始化一致。

### 2) `fastBPE` 报错怎么办？

部分可视化/翻译展示代码可能依赖 `fastBPE`（在某些环境默认未安装）。如果你仅关注训练与 BLEU 评估，可以先跳过相关 cell，或者改用 `subword-nmt` 的结果完成解码/后处理。
