# 第一周主要学习内容发言稿

## 第 1 页：封面

老师好，本周汇报的主题是“第一周主要学习内容”。本周学习以 Stanford CS336: Language Modeling from Scratch 为主要材料，围绕语言模型的输入表示、计算过程、训练成本、优化机制和 Transformer 架构展开。

整体目标不是介绍课程背景，而是汇报第一周形成的技术框架：从文本如何进入模型，到张量如何参与计算，再到训练过程如何更新参数，以及现代 Transformer block 由哪些模块构成。

## 第 2 页：本周学习内容概览

本周内容可以概括为六个部分。

第一部分是语言模型的输入与预测链路，重点是从文本到 token、embedding、logits 和 loss 的完整路径。第二部分是 Tokenization 与 BPE 方法，重点是文本切分和词表构建。第三部分是 PyTorch 张量计算，重点是 batch、seq、hidden、vocab 等维度的流动。第四部分是 FLOPs 与训练成本分析，重点是计算量、显存、带宽和硬件利用率。第五部分是损失函数、梯度和参数更新机制。第六部分是 Transformer block 的模块化结构及现代 LLM 的常见架构变体。

阶段性结论是：目前已经形成“文本表示、张量计算、训练成本、优化过程、架构模块”这一条基础知识链路。

## 第 3 页：语言模型输入与预测链路

这一页展示语言模型从输入文本到训练目标的基本计算链路。

语言模型并不是直接处理自然语言字符串。文本首先经过 tokenizer，被映射为 token IDs；随后 token IDs 通过 embedding 表转换为连续向量表示；这些向量进入 Transformer 后，模型输出 logits，也就是对词表中每个 token 的预测分数；训练阶段再通过 loss 衡量预测分布和目标 token 之间的差异。

这里的关键结论是：token 是训练和推理过程中的基本离散单位。自然语言层面的“词”并不等同于模型内部的 token，后续所有计算都建立在 token 序列及其向量表示之上。

## 第 4 页：Tokenization 与 BPE 方法

这一页讨论 Tokenization 与 BPE。

Tokenization 解决的是文本到整数 ID 的映射问题。由于模型不能直接处理字符串，tokenizer 需要先将文本切分为 token，再将 token 映射到词表中的 token id。

BPE，也就是 Byte Pair Encoding，可以理解为一种基于频率的子词合并算法。它通常从更细粒度的字符或字节片段开始，反复合并语料中出现频率较高的相邻片段，最终形成包含常见词、词根、标点、代码片段等元素的词表。

这一部分与后续 Assignment 1 直接相关，因为作业会要求实现 BPE tokenizer。也就是说，后续需要将这里的算法流程落实为可测试的代码。

## 第 5 页：张量形状与模型计算表示

这一页讨论 PyTorch 张量形状。

语言模型中的计算不仅是调用 API，更重要的是明确每一步张量的 shape。比如输入张量可以表示为 `batch × seq × hidden`，其中 batch 表示批大小，seq 表示序列长度，hidden 表示每个 token 的向量维度。

当权重矩阵形状为 `hidden × vocab` 时，通过矩阵乘法可以得到 `batch × seq × vocab` 的 logits。这里 vocab 维度对应词表大小，即模型对每个位置预测下一个 token 的分数。

einsum 的价值在于，它把参与计算的维度名称直接写出来，使得张量运算的输入、输出和求和维度更加清晰。没有出现在输出表达式中的维度会被求和。

此外，本周还建立了基本的显存估算直觉：float32 占 4 bytes，float16 和 bfloat16 占 2 bytes。因此，仅保存一个 70B 参数模型的 bf16 权重就约需 140GB 显存。

## 第 6 页：FLOPs、带宽与训练成本分析

这一页讨论训练成本。

模型训练成本不能只用 GPU 数量描述，还需要分析计算量和数据搬运。FLOPs 表示完成一次计算所需的浮点运算数，FLOP/s 表示硬件每秒能执行的浮点运算数，MFU 则衡量实际算力利用率相对于理论峰值的比例。

另一个关键概念是 arithmetic intensity，即 FLOPs 除以 bytes。它描述每搬运一个 byte 数据所能完成的计算量。

如果算术强度较低，计算过程中大量时间花在读取和写入数据上，这类操作通常是 memory-bound。例如单独执行 ReLU 时，每个元素只做很少计算，但需要读写大量数据，因此容易受内存带宽限制。相比之下，大矩阵乘法的计算密度更高，更可能接近 compute-bound。

这一部分的结论是：理解训练效率时，需要同时考虑 dtype、batch size、FLOPs、显存、带宽和 MFU。

## 第 7 页：损失函数、梯度与参数更新机制

这一页讨论训练循环。

训练过程可以分为几个步骤。首先是 forward，用当前参数计算预测结果；然后通过 loss 衡量预测结果与目标值之间的差异；接着执行 backward，沿计算图计算梯度；之后 optimizer.step 根据梯度更新参数；最后 zero_grad 清空梯度，为下一轮迭代做准备。

这里需要区分参数梯度和中间激活梯度。参数梯度用于直接更新权重；中间激活梯度本身不是最终要更新的对象，但它是链式法则中的必要中间量，用于把 loss 的影响继续传递到前序参数。

因此，反向传播不是单纯“求参数梯度”，而是沿计算图系统地传播导数信息。

## 第 8 页：Transformer Block 的模块化结构

这一页讨论 Transformer 的模块化结构。

现代语言模型的基本单元可以拆解为几个核心模块。Self-Attention 负责基于上下文重新聚合 token 表示，使每个位置能够利用序列中其他位置的信息。MLP 或 FFN 对每个 token 的表示进行非线性变换，提升模型容量。Residual connection 保留输入路径，有助于训练深层网络。Normalization 用于稳定激活尺度，提高训练稳定性。

因此，Transformer block 并不是一个不可拆分的整体，而是由多个可实现、可测试的子模块组成。这一点对后续代码实现非常重要。

## 第 9 页：现代 LLM 架构的主要变体

这一页讨论现代 LLM 架构中的常见变化点。

不同模型虽然都可以被称为 Transformer-based language model，但具体实现可能差别很大。激活函数可以从 ReLU 演变为 GELU 或 SwiGLU；位置编码可以使用 sinusoidal encoding，也可以使用 RoPE；归一化方式可以采用 LayerNorm 或 RMSNorm；attention 结构也可能从标准 multi-head attention 变为 MQA、GQA 或 MLA。

此外，模型 shape 相关的超参数也非常关键，包括 hidden dimension、层数、attention head 数和 FFN dimension 等。

因此，阅读模型报告或配置文件时，不能只关注参数量，还需要关注 block 结构、attention 变体、位置编码和主要超参数。

## 第 10 页：后续工作

最后一页是后续工作计划。

下一阶段将以 Assignment 1 为实践主线，把本周学习的概念落实为代码。第一步是实现 tokenizer，重点包括 BPE 的 encode、decode 和 merge。第二步是实现基础模块，包括 Linear、Embedding、Loss 和 Optimizer。第三步是实现 Transformer 相关模块，包括 Attention、RoPE、FFN 和 Norm。第四步是运行测试并训练 tiny language model，记录实验结果。

阶段总结是：本周已经完成语言模型输入表示、计算成本、优化过程和架构模块的基础梳理；下一步需要通过作业实现验证这些概念是否真正掌握。
