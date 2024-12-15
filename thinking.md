# Intro

该文档记录一下自己在编写项目时一些思考

落笔于2024-11-20 16：30

# 2024-11-20 关于从原始数据到手动计算的特征处理

经过简单预处理后的数据存储在 `'/data/Ruiwen/data_with_ICA.pkl'中，包含 `'eeg'、'eye'、'au'、'label'、'subject_list'`，Info如下：

```python
dict_keys(['label', 'subject_list', 'ch_info', 'info', 'eye_info', 'eeg', 'eye', 'au'])
EEG: 1-70filer, 50Hz notch, With ICA, 256Hz resample;
Subject : 1-34 subject, no 1,23,32 subject, 15 subject exists eye data missing; 31 person, 48 question, 31 channel;
Labels : 0: Confused,1: Guess, 2:Unconfused, 4: Think-right;
Eye Track data : Pupil diameter left, Pupil diameter right,Gaze point X, Gaze point Y, Eye movement type, Gaze event duration
['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']
[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34]python

```

> 注意这里没有将AU的信息记录在Info中，后续进行补充。

截至到现在已经编写完加载简单预处理后的原始数据后，对于脑电数据，经过滑窗并计算DE特征，已经用GPT重构代码使其更加工程化。考虑到每次加载时间比较长，所以采用一次加载后就存储保存的形式，后续只需要加载保存好的（一个缺点就是每次保存后的名字是一样的，所以如果其中受试者或频段什么发生改变后，但无法辨识情况下，会加载旧的）（考虑一个解决办法为用名字来唯一标识，后续再说，现在一般也不动）

接下来在写眼动加载的部分代码，由于读取的是原始数据，需要编写眼动数据手动计算特征的代码。

即步骤为：

- 加载数据
- 滑窗
- 手动计算特征，包含41个特征点

后续编写人脸部分的代码同上操作

17：19更新：重复上述步骤有点麻烦，压力有点大，就先按照之前方式，直接加载预先处理好的数据来用

人脸特征，唔，其实存储的原生数据已经是计算后的特征了，而不是原始AU强度点，算了，不管了，直接跟眼动一并处理

17:30更新：已经完成对特征加载代码部分编写，编写于LoadFeatures.py下，接下来将读取到的特征数据，封装成torch.Dataset形式，然后划分训练集和测试集，训练集和测试集划分还是先以留一人交叉验证方式（后续补充跨被试实验，用于丰富文章内容）

# 2024-11-21 20：15，封装成torch.Dataset形式

需要加Norm，但是Norm加载哪儿，需要考虑：

特征上去做Norm，每个特征单独去做Norm
做Zscore-Norm以去除个体差异性，做Min-Max Norm以缩放到-1，1区间内。
如果是在每个人上去做，这样以去除个体差异性

在处理数据时发现一个特点，即在做DE特征提取时，已经在一个通道上一个频带内做了Min-Max归一化，并且对于不满足15s的数据自动补齐数据，用0填充。

如果这时候对其再在每个人内部做Zscore标准化，emmm，会有影响吗？（按之前的训练经验来看是有效的，能够提升准确率。当然也可能是对其他特征做了归一化产生的效果）。TODO

关于脑电数据的补0考虑，因为Ruiwen实验中，每道题目最长时间为15s，（从佳宝CAL得出）

所以最长补充到15s即可

之前脑电形状为（50592，150），即（34\*31\*48，5\*30）

新版本：（34\*48，31，5\*15）

这里的TorchDataset就要做区分训练集和测试集了，那么提前一步，在计算特征时就根据人划分训练和测试，还是说在提取特征中间再加一个模块，用于从提取的特征划分训练集和测试集？用第二个模块更加合适

新模块SplitDataset：用于根据受试者ID获取指定的特征

而FeatureDataset：只需要传入SplitDataset类的数据，就可以将其封装成可供训练和加载的Dataset类

之前的写法是耦合在Dataset中了，因为是留一人交叉验证方式，以及十折交叉验证方式。
（明天再说吧，欸嘿）（划掉）继续干

按Copliate给出的代码，其为个人：特征的字典，算了，还是参考之前的方式去做

考虑一下此时各个特征的形状：

脑电：(1488, 31, 5\*15)
眼动：(1488, 41)
人脸：(1488, 115)
标签：(1488, )

23：41，逻辑通畅了，明天检查一下是否有BUG，然后就可以着手搭建模型了，争取这周修改一版模型，然后训练出一个结果出来。

# 2024-11-25 10：33，检查Dataset逻辑，搭建后续逻辑代码

首先检查撰写的Dataset代码，对于非跨被试和跨被试实验，数据集划分逻辑上是否合理

check done！非跨被试划分无问题，跨被试实验亦无问题。

#TODO：对于2分类任务中，容易出现类别不均衡情况，或者测试集过少情况，后续考虑下

那么当前数据的加载逻辑写完了，下一步是模型搭建，再下一步是训练过程，下一步是测试过程，最后是日志系统，和绘图系统，以及自动化脚本系统。

真的是，一整天，被学术讲座材料给消磨光了，一点都没进度，晚上再rush一把吧

20：30：完成Models部分，以之前写的MAFFM框架（后续再起个名字）

init_weight这种初始化权重方式在开源库中都会存在，说明存在理论支撑在其中

先起名字吧，不然模型不好起名，由AFFM+Transformer结合而出，AFFM名字为Adaptive Feature Fusion Model

多模态注意力特征融合情绪识别策略：Multi-Modal Feature Attention Fusion Encoder strategy

然后将模型模块化，包含特征提取模型、特征对齐模型、特征融合模型、特征编码模型、分类模型

对于模态融合模块，如果是只有两个模态两两融合那么只有一份参数可以理解

但是对于超过两个模态的情况，如果只用同一个自注意力层进行融合，产生的参数共享，对于不同模态之间注意力信息交叉会产生混淆

解决办法，三种策略：

1）正交顺序融合：EEG融合Eye形成中间态Mid，Mid融合Au；后续添加L1和L2正则化以正交。

2）多个attn：EEG与Eye融合；EEG与Au融合；Eye与Au融合；最后三者拼接。

3）共享的attn：一个attn，其中Eye作为query，与EEG融合；Au作为Q，与EEG融合。

4）现在的最莫名其妙的策略：一个attn，EEG+Eye，EEG+Au，Eye+Au。保留了EEG丰富信息，同时EEG指导

上述融合策略，一个问题即：当EEG完全缺失状态下，真的OK吗？一个想法，Eye、Au作为Query，指导生成EEG中间态表示或融合状态表示，用于后续深度编码（未成熟想法）

之前融合策略中，即上述第4个，EEG作为Q，简而言之，保留了Eye和Au中与EEG相似度最大的特征值，同时保留了两份EEG信息，保留了Eye、Au的最突出的特征信息，保留了Eye特征以及Au中与Eye最相似的特征以及Au最突出特征信息。

这么来看的话，第三步Eye+Au融合意义不大。以及需要加些线性变换来整合信息，之前没有整合。

因此上述策略，需要跑下实验看看效果，预计跑1、3、4，4为ori，但是添加线性变换和移除Eye+Au。

对于EEG完全缺失情况，上述特征，无法得出有效的推导出EEG或受Eye和Au指导的融合信息（唔，加掩码层试试？后续考虑吧，先把上述1、3、4实验跑出来看看效果）

还有在这儿，因为将一整个trial变成了一个特征，那么没有时序信息，那么引入多个Seq以表示不同的子空间表示，拓宽表示空间。

上述4为最终采用的第4版融合策略，后续模态缺失实验是在第一版上实验的。

因此现在还原第4版实验，但是改用EEG+Eye，EEG+Au，同时这边需要再做一组对照实验为Eye+EEG和Au+EEG，即Eye和Au指导EEG（这组实验条件较弱，可以不做）

暂且不考虑单模态情况，只考虑三模态情况，先写的耦合一些

如何避免耦合度来用GCN提取脑电特征是一个问题，TODO

后续再优化吧，先只搭了个骨架。

# 2024-11-26 17:16，跑通模型，完成训练部分搭建

先完成模型搭建部分，以简单快速高效为主来完成，尽快跑起来实验，时间不多了。

配置文件以之前学到的那个自动加载配置文件方式来做，虽然忘了是什么了

21:30：可以打印网络，但是网络输入会存在问题，因为存在需要输入邻接矩阵和图结点，这个在后续编写的时候再做补充

接下来写Trainer部分，用于调用模型进行训练环节。

一个Trainer类往往包含：训练、推理、日志记录三部分

Trainer构建完了，接下来还差Logger、Schedule（可选）、Arguments、Main四个模块，以及shell，哦对，还一个Metrics模块（Acc、F1-score、CM），以及Visualize模块

接下来的任务，构建完Logger模块后，搭建Main逻辑，然后跑通训练

# 2024-11-27 11:13 搭建Main逻辑，跑通训练

哪那么多事，干就完了，多干干总没错，干完一样是一样

昨天的Train.py类可能存在问题，但是问题不大，这个在后续运行Main函数时逐步去看

12：29：main.py的主要逻辑差不多了，后续是在调试中完善，然后config.yaml格式有点问题，不能适配不同的数据集以及不同的模型，需要做点可扩展

然后还有可视化模块的实现，可视化准确率，混淆矩阵，loss和acc曲线在tensorboard中显示了

接下来，下午的任务是打通整个训练链路，跑起来再完善。

22:08：打通训练逻辑，先训练上

eeg：（128，31，75）（这里忘了将后两维度合并了）（哦，这里在后面提取eeg图特征时改变形状了）
eye：（128，41）
au：(128，119)

targets：128

之前脑电为（750，31，150），那么传入到网络中应该为（128，31，150），INPUTS_DIM为（160\*3\*2，41，119）

那么输入的维度为75，即脑电特征的维度

还要加上Norm，不然无法消除量纲的影响

欸，融合部分改写下，TODO
x_1 = self.attention(eeg, eye) + self.MI(eye, eeg)

x_2 = self.attention(x_2, au) + self.MI(au, x_2)

x_3 = concat(eeg, x_2)

混淆矩阵和Norm后续再考虑吧，反正现在能训起来了

# 2024-11-28 10:15，跑通了，添加Norm，优化日志，添加可视化和结果处理

一个完整的训练流程跑通了，调整batch为64，训的还蛮快

batch为64时，训练丢弃的数据还蛮多，如果效果好就不管他，如果不好就也要考虑这块儿因素

没有输出最终结果，不太好

不同人的tensorboard可视化需要加以区分

优化了日志和checkpoints的输出

config有点冗余，优化一下。已优化，适配不同模型配置和数据集配置，后续如果添加其他类型数据集或模型再修改代码。

readme不重要，上传到git以实现版本控制而已，避免编写记录丢失，2024-11-28，11：18第一次提交。

正则化处理，现有的代码跑上，把想跑的几个对比实验列一下，分析一下是否有跑的必要，设计一下自动化，然后训练一下，先跑着，在跑的同时去写下论文或者研究一下模态缺失算法等其他的。

手动提取的特征，直接去对整体做归一化存在问题，还是需要对每类特征做归一化

据我所知，眼动的话，是从3-5种眼动特征点去手动计算的41条特征，因此直接对每一种特征去做归一化，来消除量纲的影响。至于个体差异性，也要加以考虑。

人脸的话，是从17个AU强度，计算的7个特征，即119个特征，这个该怎么归一化？考虑直接些，每个特征点去做独立归一化。

Norm逻辑传参错了可还行，已修改。

写点文章，然后把实验跑上，以及列下要跑的实验种类，跑实验要巨长时间，可以先跑跨被试，这个会快些

跑个实验要两个半小时，跨被试还存在问题，调下吧

 修复了graph逻辑，修复了跨被试实验，添加了正则化

 保存错了，保存成最小的了，17：05，重新跑吧，大概晚上7.30出结果

 4分类的任务也得添加上，理论上来说应该可以直接跑通。

 20:41,光编码了，有点乱，我要的是什么？

 首先是还原之前的结果，最终我应该拿到的输出结果为，每个人测试集上的最高准确率，以及其平均准确率，输出结果应该输出到一个文本文档或表格中加以记录TODO

# 2024-12-03 18:52, 补全输出配置和结果到统一文件的管理

 拖得有点久了，这个不是很难的工作，直接完成，然后把训练任务跑上吧，然后再在公开库进行算法检验，这个将其数据提取做好就行。

 首先是配置文件的逐级拆解，设计为均拆解成一列的形式进行存储。

 然后修复Std的计算逻辑

 20:33，已修复，费时1.5h，hh，有点浪费时间了，测试一波

# 2024-12-04 11:25 修复保存历史训练记录的bug，开启大规模实验

 修复了保存config和metric到同一csv文件便于后续统一查看和管理，测试中

 每次都要创建一个excel文件，这不大行

 考虑的场景也就是在跨被试和非跨被试实验中，会考虑不同人的情况，不行的话分别定义一个文件来做记录（再调试一下就看看当前这个情况）

 评价：6，因为每列的类型不一样，csv中所有均为str，而新构建的df中会存在int类型，equal比较时也会比较类型

 测试通过便训练大规模的各个模态融合的情绪识别实验

 大规模实验受限于CPU的限制，明天迁移到其他机子上再跑一些公开库的实验（这部分结果缺失）

# 2024-12-05 11：40 竟然能跑完欸

 昨天晚上20:30跑的，今天11：40发现已经到30个人了，那么12点能跑完，也就是说总共跑了12+3.5=15.5个小时，也还行。还行个damm，数据量少的单模态跑完了，数据量多的比如说三模态，才跑到第15个人，这得跑到什么时候，今晚12点或者明早才能看到结果，可能程序有问题，但是不想调了，多启用几个机器在跑吧

 得需要把公开库的代码逻辑补全它，然后让他跑上，毕竟要跑至少三个公开库，分别是HCI、DEAP以及SEED，高天师兄应该有预处理好的数据

 17:35，跑了21个人了，难蹦，预计时间需要24h以上，所以早点把其他实验代码写完然后在后台跑着去吧。

# 2024-12-06 寻找的新的解决策略，在公开库上的评测

 新的结果出来了，很难崩，脑电的准确率比三模态融合还要高一些

 改动上，Encoder层，改用了LayerNorm，

 上述情况，再多看看论文来优化下算法再说吧，先去完善公开库实验

 先是HCI库，处理HCI库的数据保存成pkl形式

 分析Raven库的数据集构建过程，对于眼动和人脸特征的提取环节省去了，直接用的现成的特征数据来做的，也就脑电的数据进一步计算了DE特征，那么HCI库高天师兄也同样做了处理，那么去看看

 参考"/home/yihaoyuan/WorkSpace/hci_workspace/graph/Train/main.py"的代码，其中包含HCI、MPED、SEED、SEED_IV（没DEAP的话，可以参考这个路径"/home/yihaoyuan/WorkSpace/MultimodelProject/multimodel_train.py"，里面有对DEAP的处理。

满抽象的，自己去写吧

HCI库的脑电数据和PPS数据放在了一个文件中，不太适合做ICA，就不做了，预处理过程于Raven保持一致

预处理过程略微有点复杂，后续再说TODO，先优先搞定利用已经处理好的数据来做实验。

[1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30]

感觉需要面向数据集特定的数据加载类，不然不同的数据集组织形式不太一样，比如说Raven是eeg+eye+au，31通道，48道题目

HCI是两类情绪arousal、valence，32通道，eeg+eye+pps特征.

de_features维度：(15540, 585)，之前raven de_features维度为：(50592, 150)，现在de_features维度为(50592, 75)，与输入到网络的维度密切相关

eye:(24, 20, 38)，pps:(24, 20, 230)

HCI有分类有两个维度，分别是arousal以及valence，分别对应不同的标签，为420，标签均为{0，1，2}，即三分类

```
Labels :  
    Arousal: 0:[\"Sadness\", \"Disgust\", \"Neutral\"]、
            1:[\"Joy, Happiness\", \"Amusement\"] 
            2:[\"Surprise\", \"Fear\", \"Anger\", \"Anxiety\"]\n\

    Valence: 0:[\"Fear\", \"Anger\", \"Disgust\", \"Sadness\", \"Anxiety\"] 
            1:[\"Surprise\", \"Neutral\"] 
            2:[\"Joy, Happiness\", \"Amusement\"]\n\

```

保存数据于"/data/MAHNOB/hci_data.pkl"

```python

data = {
    "eeg": hci_de_features,
    "eye": eye_features,
    "arousal_label": arousal_label,
    "valence_label": valence_label,
    "subject_list": subject_lists,
    "ch_info": ch_info,
    "info": "EEG: 1-70filer, 50Hz notch, 256Hz resample;\n\
Subject : 1-30 subject, no 3,9,12,15,16,25 subject; 24 person, 20 trial, 32 EEG channel, 7 pps channels;\n\
Labels :    Arousal: 0:[\"Sadness\", \"Disgust\", \"Neutral\"]、1:[\"Joy, Happiness\", \"Amusement\"] 2:[\"Surprise\", \"Fear\", \"Anger\", \"Anxiety\"]\n\
    Valence: 0:[\"Fear\", \"Anger\", \"Disgust\", \"Sadness\", \"Anxiety\"] 1:[\"Surprise\", \"Neutral\"] 2:[\"Joy, Happiness\", \"Amusement\"]\n\
Eye Track data : DistanceLeft，PupilLeft，ValidityLeft，Distance Right，Pupil Right，Validity Right，Fixation Index，Gaze Point X，Gaze Point Y，Fixation Duration\n\
     PPS data : ECG, GSR，Resp，Temp，Status",
}

```

64*585，de的特征读取存在问题

# 2024-12-07 解决HCI bug，跑通实验

## HCI bug修复，跑通实验

想起来了，Raven库的DE数据是重新旋转到了（N，ch_nums, samples）大小，那么HCI库也做相应的处理

真不知道高天当初de是怎么存储的，重新保存下HCI的原始特征吧，然后自行计算算了

24个人，20个trial，32通道, 每个trial的sample数不一样，但这个正常，直接保存下来即可，保存成和Raven一样的数据存储形式'(per_idx, n_trials, n_samples, n_channels)'

HCI能跑通了，不知道结果会怎么样，考虑看看其他的数据集，多跑几组实验。

# 2024-12-08 处理DEAP、添加模态缺失实验

继续对DEAP数据集进行处理，处理成可供训练的形式

在模型训练过程中，训练集是有效的，但是测试集上准确率偏低，表明了存在过拟合现象，后续需要围绕过拟合开展操作。同时的话多模态融合的效果差异不明显，需要改进融合网络。（尽快完成其他部分撰写，然后再考虑优化模型的事情）

今日要做项：

* 整理HCI的结果
* DEAP数据集处理和训练
* 跨模态的实验代码和跑通
* 论文内容撰写，撰写第4和第6章
* 考虑绘图的代码，绘制一些图，这个在撰写论文中设计要放什么图

## DEAP数据集

DEAP数据集可以搞一下，在/data/zhanggt/DEAP下，有eeg和pps特征，并且pps特征已经被提取出来了。

参考/home/yihaoyuan/WorkSpace/MultimodelProject/deap_test.ipynb来处理deap原始数据，处理代码存放在/mnt/nvme1/yihaoyuan/Raven/RavenEx/multimodal_emotion_recognition/raw_data_process/DEAP路径下。

DEAP数据集描述：https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html

DEAP的最长的长度为61.

used_participant_no = list(np.arange(1, 33))

预处理后的DEAP数据集包含脑电和PPS数据，被下采样至128Hz，去除了EOG伪影，4-45Hz带通滤波，分割成了60s，移除了前3s的基线，共40个通道，其中1-32为脑电，33-40为生理信号，

这么来看的话，pps数据为原始数据，没有经过特征提取，可以找下之前的特征提取代码，这块儿还是我写的，不一定对可能。

pps数据包含：

33-34：EOG；35-36：EMG；37：GSR；38：RSP；39：体积描记器；40：TMP

然后还有脸部视频数据，可以提取Au特征，然后计算，但这块儿代码没找到，数据也没找到。

分析了一下，现有的数据只有预处理后的数据，可以提取DE特征，关于PPS特征和AU特征短期内做不了，所以暂且搁置，或者直接拿预处理后的数据做实验，但这个优先级下降

新的优先级任务：跨模态实验的完成

## 补全跨模态实验

实验思路：预训练好的模型提供融合信息，融合信息作为主导模态弥补脑电模态缺失的不足，完成仅需要眼动和人脸就可以实现的情绪识别。

融合信息的获取：AFFM提供的，或者Transformer编码后提供的，两类做个消融实验。

可供提供的绘图，主导模态融合后与其他模态的注意力分数，融合后的信息，与之前脑电信息的相关系数评估（再考虑）

接下来，看看论文来获取一下更多的思路吧。

梳理一下自己的实验思路：

1. 设置为跨模态实验，标记脑电模态缺失
2. 加载预训练好的模型，根据已有的数据，从中取出一小批数据，送入到预训练网络代表中间状态，
3. 然后根据中间状态去结合其余的假设脑电数据缺失状态下的数据继续训练，用于模拟脑电模态缺失，
4. 然后在预测过程中，同样结合这一批数据取得已经预训练好的网络的中间融合状态，用于在测试集上融合预测。

难点：

1. 加载预训练好的模型
2. 小批数据的设置大小，暂定64，抽取数据采用随机采样的方式（这里也可以考虑下类别均衡性，即对每个类别加权来抽取数据，或者用过采样方式），然后固定这批数据，在每次训练和推理过程中都是采用这批数据（要么复制，要么替换）

# 2024-12-09 设计下消融实验，集中在模态融合策略上

现有的融合策略为主导模态与其他模态的融合，接下来测试下完全融合策略的效果，以及测试下不同的语义扩展大小对于融合效果的影响

已经添加了三类融合策略的消融实验

接下来验证语义扩展的消融实验研究，即seq_len多大会比较合适（感觉这个与特征的大小有关系，当特征多的时候，更大的语义扩展能更好的表征特征，然而较小的特征数量若进行较大的语义拓展容易产生噪声干扰，比如说pps的影响），既然说seq_len * d_model会产生影响，那么d_model设置一个较小的值，emmm，但这样的话对于融合以及编码模块的d_model会变小，所以可能探究seq_len以及d_model的最佳配比组合。因为后续融合以及编码模块的d_model受d_model以及seq_len的影响，所以要改需要一起改。

所以得出来的结论为探究d_model以及seq_len的最佳配比.

## TODO

（虽然知道要做，但还是有点不想做qwq，这个关于消融seq_len的设置，等写完跨模态任务后再做这块儿吧）

# 2024-12-11 消融seq_len的最佳设置

采用渐进式搜索策略，考虑到参数量，计算时间影响，（待会儿画个图）

1. 固定d_model 搜索seq_len
2. 找到合适seq_len后，小范围搜索d_model
3. 精细化进一步搜索

注意设置seq_len考虑输入的维度大小，确保d_model * seq_len要大于最大的特征大小，即大于960，即当d_model为160时，seq_len至少大于6

在高天的搜索中,d_model设置为160为最合适，那么这里先固定d_model为160，去搜索seq_len。

后续再去考虑d_model搜索不受高天论文的影响，进行更大范围的搜索。

seq_len设置搜索范围为 `6，8，10, 12，16`，这下对于Arousal和Valence将会有 `4*2=8`组实验，显存和cpu占用还是可以接受的范围，CPU已经拉满了qwq，那么一次实验最好不要超过10组

## 跨模态代码的编写

按照大论文的思路去编写这部分的实验代码

代码逻辑为定义两个模型，一个用于训练，一个用于

这代表性数据，用什么逻辑去获取呢，是以类的形式表示，还是以中间变量形式表示？

当前模态缺失模型接受的其他输入均不变，唯一变动即EEG的输入改成了预训练模型的中间输出，也就是说拿到的输入向量为 `（B，seq_len，d_model * 2 * 3 ）`，其维度与其他模态维度不同，所以也要经过Embrace模块来对齐模态

HCI库跑完一次实验需要8个小时。

时间不多情况下，还是以面向过程的思想快速实现代码逻辑，直接在另一个代码空间内去改写适配跨模态任务，快速跑通为好，同时避免干涉到主任务。

加载数据时逻辑改写一下，模型计算逻辑改写一下，其他部分不需要改动就可以了。

添加一个加载checkpoints功能函数。

TODO：对齐的设置中，投影维度大小设置需要考虑，既要避免信息冗余，也要避免信息丢失。

寻找一小批代表数据时，避免测试集暴露，所以在划分测试集和训练集后

所以在加载测试集时，同时加载

加载了checkpoints后也要初始化模型，但不需要初始化优化器

模型的keys是个有序字典，

```


dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict'])
feature_extract.feature_global.gcn1.weight torch.Size([585, 160])
feature_extract.feature_global.gcn1.bias torch.Size([160])
feature_extract.feature_global.gcn2.weight torch.Size([160, 160])
feature_extract.feature_global.gcn2.bias torch.Size([160])
feature_extract.feature_global.gcn3.weight torch.Size([160, 160])
feature_extract.feature_global.gcn3.bias torch.Size([160])
feature_extract.feature_global.pool.attn_gcn.weight torch.Size([480, 1])
feature_extract.feature_global.pool.attn_gcn.bias torch.Size([1])
feature_align.docking_0.weight torch.Size([1600, 960])
feature_align.docking_0.bias torch.Size([1600])
feature_align.docking_1.weight torch.Size([1600, 38])
feature_align.docking_1.bias torch.Size([1600])
feature_align.docking_2.weight torch.Size([1600, 230])
feature_align.docking_2.bias torch.Size([1600])
fusion.attention.q.weight torch.Size([16, 160, 1])
fusion.attention.k.weight torch.Size([16, 160, 1])
fusion.attention.v.weight torch.Size([16, 160, 1])
fusion.attention.up.weight torch.Size([160, 16, 1])
fusion.fn.weight torch.Size([960, 960])
fusion.fn.bias torch.Size([960])
attention_encoder.encoder_layer.0.norm_1.alpha torch.Size([960])
attention_encoder.encoder_layer.0.norm_1.bias torch.Size([960])
attention_encoder.encoder_layer.0.norm_2.alpha torch.Size([960])
attention_encoder.encoder_layer.0.norm_2.bias torch.Size([960])
attention_encoder.encoder_layer.0.attn.q.weight torch.Size([120, 960, 1])
attention_encoder.encoder_layer.0.attn.k.weight torch.Size([120, 960, 1])
attention_encoder.encoder_layer.0.attn.v.weight torch.Size([120, 960, 1])
attention_encoder.encoder_layer.0.attn.up.weight torch.Size([960, 120, 1])
attention_encoder.encoder_layer.0.ff.linear_1.weight torch.Size([2048, 960])
attention_encoder.encoder_layer.0.ff.linear_1.bias torch.Size([2048])
attention_encoder.encoder_layer.0.ff.linear_2.weight torch.Size([960, 2048])
attention_encoder.encoder_layer.0.ff.linear_2.bias torch.Size([960])
attention_encoder.encoder_layer.5.norm_1.alpha torch.Size([960])
attention_encoder.encoder_layer.5.norm_1.bias torch.Size([960])
attention_encoder.encoder_layer.5.norm_2.alpha torch.Size([960])
attention_encoder.encoder_layer.5.norm_2.bias torch.Size([960])
attention_encoder.encoder_layer.5.attn.q.weight torch.Size([120, 960, 1])
attention_encoder.encoder_layer.5.attn.k.weight torch.Size([120, 960, 1])
attention_encoder.encoder_layer.5.attn.v.weight torch.Size([120, 960, 1])
attention_encoder.encoder_layer.5.attn.up.weight torch.Size([960, 120, 1])
attention_encoder.encoder_layer.5.ff.linear_1.weight torch.Size([2048, 960])
attention_encoder.encoder_layer.5.ff.linear_1.bias torch.Size([2048])
attention_encoder.encoder_layer.5.ff.linear_2.weight torch.Size([960, 2048])
attention_encoder.encoder_layer.5.ff.linear_2.bias torch.Size([960])
attention_encoder.fn.0.weight torch.Size([160, 960])
attention_encoder.fn.0.bias torch.Size([160])
classifier.classfier.1.weight torch.Size([3, 1600])
classifier.classfier.1.bias torch.Size([3])

```

有点意思哈，分析权重分布，可以发现浅层的encoder拥有较为尖锐的分布，参数值较大，而深层的参数，更好保持模型互补性，所以选用深层的FNN输出可能会更好些

attn后有一个fn层，用于映射到embed维度上

看热图并不能看出来什么特别的规律出来，不管是浅层还是深层都具有较大的激活值，模型都处于激活状态，相对来说深层的看起来更加稀疏一点。算了不管了，继续设计实验。

看起来今天晚上把跨模态实验是跑不完了，那就去做一下消融实验超参搜索去。

# 2024-12-12 完成跨模态实验代码

今天必须得把实验代码写出来并且跑上，不然时间真不够用了，后续还有消融实验要跑，还有绘图要画。

14：49，还在跑消融实验中，占用CPU过多，不太适合去写代码，先去撰写论文，等消融实验跑完再来继续编写跨模态实验代码，大概要4-5个小时后看训练进度来看qwq，好久哇，后续还得再重新跑一遍Raven多模态融合结果，先去写论文吧

# 2024-12-13 真的要完成了跨模态实验代码

昨天实验跑的有点久，所以今天集中把实验代码完成，然后跑个结果，我承认有赌的成分，结果不好再另说，单眼动的结果过高了qwq

上次任务进度：集中在Dataset中，处理缺失任务的数据

拿到中间输出后：

print(result.shape)torch.Size([63, 3])

print(fused_features.shape)torch.Size([63, 10, 960])

print(encoded_features.shape)torch.Size([63, 10, 160])

后面两个可以作为直接的数据输入输入到AFFM中进行融合

注意获取标志性数据是否需要对其进行打乱顺序，其实打不打乱都可以

后续在脑电模态缺失情况下，模型接受的脑电输入仅为这一组代表性输入的随机采样。

如果说这样取得一小批代表性数据的中间输入作为解决方案，那么为什么不是从原先训练过程中随机一小批数据来作为缺失输入呢？（这也作为一类消融实验，即一小批代表数据的来源）

重构Model，创建新的Model专门适配当前模态缺失的场景的模型

现在做的是直接拿到encoded_features作为融合的输入，那么形状为[63, 10, 160]，为了统一一下还是送入到特征对齐模块中进行特征对齐。

那么特征对齐模块接收的输入为N，D

模态接受输入，miss_data,代表缺失的模态，主要替换脑电模态，其他操作正常进行，而在特征提取后，将脑电模态数据拼接上去。

但是需要考虑Embrace中的对齐形状会发生改变，这个后续初始化缺失模型时，再做指定

假设继续从上述实验思想中继续模拟现在这个情况，那么现在存在问题即标签不匹配的问题，那么采用从小搓数据中随机采样对应的数据构成新的数据。还有一种思路，即只需要从原有数据集中随机采样类别数目的数据，这样能够借助这部分数据提供一种感知能力，来指导模型学习，即这种代表模态缺失。

代表性数据来自于训练集，那么设置测试集时，代表性数据同样应该来自于训练集。那么这块儿，该怎么做呢

创建一个类，代表一小撮数据，后续取代表数据，构建一个函数，用于传入输入的label，然后根据label从小搓数据中随机抽取对应的数据

明天上午能把代码写完，实验跑上，然后开始写论文！！！

任务存档：MissTaskData的编写，重新构建


# 2024-12-14 构建跨模态实验


大记忆恢复术：昨天完成到了在Dataset类中取每一个元素时，替换eeg数据为代表性的融合数据，结果因为总是卡住，所以考虑在外部完成一个batch的数据的替换处理，因此定义了一个MissTaskData，用于加载和存储这一小批代表性中间数据，并且定义一个函数处理面临一个batch数据（划掉，这样效率太慢），对划分好后的训练集和测试集数据，将其中脑电数据替换成中间数据，一次处理，正常加载。


因此现在工作流为：

1. MissTaskData构建：
   1. 从训练集中随机抽样每类代表性的为num个，共计cls_num * num个数据用于构建，
   2. 然后加载预训练模型，对于代表性数据从预训练模型中获取中间状态
   3. 存储中间状态数据和标签组合
2. MissTaskData处理训练过程数据：对于trainSet数据，以及testSet数据，将其中EEG替换成中间数据
   1. 传入data和label：根据label从已经建立好的随机抽取对应的，获取索引
   2. 根据索引获取中间数据，然后替换EEG数据？（这个，可以把GCN注释掉，或者可以考虑用GCN来做进一步的特征提取TODO）
   3. 数据处理完了，接下来是模型计算过程
3.
