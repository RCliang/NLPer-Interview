## 问题总结
### Q1： Transformer怎么来防止梯度消失
答案：在qkv计算的时候使用/sqrt（d-model）的方式来,除以sqrt(dk)的目的是使注意力得分归一化，相当于除以标准差。
### Q2：Layer Normalization和Batch Normalization的区别与机制
答案：BN是针对学习样本的整个mini-batch之间比如每个词的样本间纵向地做标准化，每次都要记录mu和方差，而LN是在一个样本的横向所有维度来做标准化。BN主要用于图像方面，LN主要用于自然语言处理。
### Q3：为什么gpt只用decoder
采用Decoder-only架构的原因主要是出于语言生成任务的需要。Decoder-only架构适用于生成式任务，如文本生成、语言翻译等，其中模型需要将输入序列转换为输出序列，因此不需要进行编码器阶段的编码，只需使用解码器阶段生成目标序列。在这些任务中，模型需要根据给定的上下文生成输出序列，而不需要对输入序列进行编码。在语言生成任务中，解码器（Decoder）扮演着核心角色，因为它负责生成输出序列，而不需要像编码器（Encoder）一样对输入序列进行编码。因此，为了更好地适应这些任务并简化模型结构，GPT类的模型都采用了Decoder-only的架构。  
优点：简化模型结构：去除了编码器部分，简化了模型结构，使得模型更易于训练和理解。更适用于生成任务：解码器专注于生成目标序列，更适用于生成式任务，如文本生成和翻译。保持模型一致性：使用Decoder-only架构可以保持模型的一致性，使得模型在不同的任务上更易于迁移和使用。

### Q4：请简单描述下self-supervised learning
### Q5: 请简单讲下TF-IDF


### Q6：AUC和F1-score的关系，样本不均衡时用哪个指标好，哪个指标会高点。 
AUC的优化目标：TPR和(1-FPR)  
横坐标：False Positive Rate（特异度） 纵坐标： True Positive Rate（灵敏度）优化auc就是希望同时优化TPR和(1-FPR)  
F1的优化目标：Recall和Precision
- auc和f1 score都希望将样本中实际为真的样本检测出来(检验阳性)，区别就在于auc除了希望提高recall之外，另一个优化目标是希望提高非真样本在检验中呈阴性的比例，也就是降低非真样本呈阳性的比例(假阳性)，也就是检验犯错误的概率，两个指标都是希望训练一个能够很好拟合样本数据的模型，这一点上两者目标相同。但是auc在此之外，希望训练一个尽量不误报的模型，也就是知识外推的时候倾向保守估计，而f1希望训练一个不放过任何可能的模型，即知识外推的时候倾向激进，这就是这两个指标的核心区别。auc希望训练一个尽量不误报的模型，也就是知识外推的时候倾向保守估计，而f1希望训练一个不放过任何可能的模型，即知识外推的时候倾向激进，这就是这两个指标的核心区别。

- 所以在实际中，选择这两个指标中的哪一个，取决于一个trade-off。如果我们犯检验误报错误的成本很高，那么我们选择auc是更合适的指标。如果我们犯有漏网之鱼错误的成本很高，那么我们倾向于选择f1score。
放到实际中，对于检测传染病，相比于放过一个可能的感染者，我们愿意多隔离几个疑似病人，所以优选选择F1score作为评价指标。而对于推荐这种场景，由于现在公司的视频或者新闻库的物料总量是很大的，潜在的用户感兴趣的item有很多，所以我们更担心的是给用户推荐了他不喜欢的视频，导致用户体验下降，而不是担心漏掉用户可能感兴趣的视频。所以推荐场景下选择auc是更合适的。

- 在不均衡样本中，实际AUC会高于f1-score 
### transformer的decoder输入中之前encoder的输出到哪个矩阵
每层 Decoder 包括 3 个 sub-layers： 

第一个 sub-layer是 Masked Multi-Head Self-Attention，这个层的输入是：
前一时刻Decoder输入+前一时刻Decoder的预测结果 + Positional Encoding。

第二个sub-layer是Encoder-Decoder Multi-Head Attention，这个层的输入是：
Encoder Embedding+上层输出。
也就是在这个层中：
Q是Decoder的上层输出（即Masked Multi-Head Self-Attention的输出）
K\V是Encoder的最终输出
tips：这个层不是Self-Attention，K=V!=Q（等号是同源的意思）。

第三个 sub-layer 是前馈神经网络层，与 Encoder 相同。
### 什么时因果推断

### XGBoost对于gbdt做了哪些改进


### python中继承多个父类的顺序

### python中装饰器的作用，作用域有哪些
### python中协程和异步编程

### 大模型压缩有哪些方法 
- 权重剪枝
- 量化
- 模型蒸馏

### gpt和GLM两种大模型的不同点
GPT采用自回归方式，基于已有序列来预测下一个元素，是生成式自回归模型。
在训练阶段，模型通过大量文本数据学习生成下一个词的能力;在预测阶段，模型利用训练好的参数来生成一段连贯的文本。
GLM主要创新了自回归填空算法：NLU任务15%的mask，遮挡采用连续多个，长度采样泊松分布。文本生成：document-level(50-100%),sentence-level(整句的15%)。还有2D的位置编码+填空序列乱序。
训练方法上：GLM基于Base模型的SFT，chatGPT是基于人工反馈的RLHF。训练Chat GPT要经过pretraining，sft，reward modeling，RLHF四个步骤。
模型衍生：![模型历史](D:\projects\NLPer-Interview\24-碰到过的问题\v2-75ef40cd622a9d91736dd73c7fe1330c_1440w.webp)
### Bert的缺点，比如掩码等，后面怎么改进的

### HMM和马尔可夫链
HMM是生成式模型，CRF是判别式模型
HMM是有向概率图模型，CRF是无向概率图模型
CRF概率归一化较合理，HMM有标签偏置的问题

### 要点
- 详细准备Transformer结构的问题
- 详细准备gpt，bert结构的问题
