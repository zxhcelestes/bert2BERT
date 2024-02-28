# 目录

<!-- TOC -->

- [目录](#目录)
- [bert2Bert描述](#bert2Bert描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [扩展流程日志](#扩展流程日志)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
    - [使用流程](#使用流程)
        - [扩展](#扩展)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# bert2Bert描述

bert2BERT模型将已经完成预训练的小型Bert模型的参数信息导入大型未训练Bert模型。使得大型模型具有小型模型的先验知识，进而减少预训练的
成本开销。Bert模型参数扩展本质是对其内部Transformer单元的参数矩阵进行扩大，按照输入输出的大小以及Transformer堆叠层数。Bert拓展可以分为广度层面和深度层面。

[论文](https://arxiv.org/pdf/2110.07143.pdf)：Chen C, Yin Y, Shang L, et al. bert2bert: Towards reusable 
pretrained language models[J]. arXiv preprint arXiv:2110.07143, 2021.

# 模型架构
广度扩展即Encoder内部计算单元的参数矩阵扩展。论文中提到两种主要方法，一是功能保留生成(Function preserving initialization, abbr. FPI )
方法,二是超前信息生成(Advanced knowledge initialization, abbr. AKI)。前者利用同层的参数信息扩展，后者会参考当前层和下一层的信息对当前层扩展。


# 数据集

使用的数据集：[SST-2](https://drive.google.com/file/d/1JmY1CuASBIA8SW-Z5wMyc9ilseU2RHDZ/view)

- 注：数据集仅用于验证扩展后的大模型embedding输出，该任务没有训练过程。

# 特性

## 扩展流程日志

参考原文，Transformer块的扩展具有先后顺序。用户可以在log文件夹中查看扩展算法流程的日志文件，以及Pytorch版和MindSpore版的比对信息。

# 环境要求

- 硬件（CPU/Ascend）
    - 使用CPU/Ascend处理器来搭建硬件环境。 由于服务器不支持torch，只对MindSpore部分内容做了Ascend适配。
  因此，涉及torch的对比试验代码请勿在Ascend调用。可调用推理代码infer和扩展代码enlarger_bert/gpt_mindspore做验证。
  这部分内容在[快速入门](#快速入门)和[推理过程](#推理过程)。
    - 如不需要Ascend计算，则正常使用提供的命令行即可。如需要Ascend，在上述只涉及MindSpore的命令行调用中，
  将device_target设置为“Ascend”。
    - 特别说明：Ascend执行速度比CPU慢,默认使用CPU计算。服务器的MindSpore版本为1.7，
  本实验的代码基于MindSpore1.9。如存在specify_prefix等关键字不存在等问题，是MindSpore版本过低的问题。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
    - [Pytorch](https://pytorch.org/)
    - [transformers](https://huggingface.co/)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.9/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/r1.9/index.html)
    - [Bert内部实现](https://huggingface.co/bert-base-uncased)
    - [MindSpore实现的Bert](https://gitee.com/mindspore/course/tree/master/03_NLP/bert#bert_for_finetunepy%E4%BB%A3%E7%A0%81%E6%A2%B3%E7%90%86)
- 运行代码相关的文件
    - 链接：https://pan.baidu.com/s/1R6ihLdBbFG130agwCuh3_Q?pwd=3kw4 
    - 提取码：3kw4 

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行扩展和评估：

在代码实现时，我们将MindSpore库中mindspore.load_param_into_net函数中的下面代码段删除。该代码段会导致强行匹配不同层的参数 ，以填补缺漏层参数的效果。本任务不需要该功能。我们在本地重写了load_param_into_net函数，而没有使用mindspore库中的该函数。需注意，功能有所差异。

```python
if param_not_load and not strict_load:
    _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load)
```

需要注意的是，本实验运行需要的ckpt文件以及SST-2数据集需要从网盘下载，或者自行生成。

下载链接如下：https://pan.baidu.com/s/1R6ihLdBbFG130agwCuh3_Q?pwd=3kw4

提取码：3kw4 

如需要自行生成，则需要运行random_init_ckpt.py或random_init_ckpt_on_gpt.py文件

运行BERT模型，需要运行random_init_ckpt.py

依次将```if __name__=='__main__':```内的内容改为
```python
    path = "./ckpt/bert_small"
    random_init_ckpt(path, "bert_small")
```
```python
    path = "./ckpt/bert_base"
    random_init_ckpt(path, "bert_base")
```
```python
    path = "./ckpt/bert_large"
    random_init_ckpt(path, "bert_large")
```
#### 注意：

在enlarger.py中对于bert_base2large的扩展方法基于的是官网下载的ckpt文件，如果要使用自己随机生成的ckpt文件，那么需要将代码进行修改。如果不希望进行修改，那么bert_base的ckpt文件需要从网盘链接中下载，除bert_base外其余文件均无影响。

修改代码PytorchEnlarger类中的bert_base2large_aki方法和bert_base2large_fpi方法修改如下（对应文件12-28行）：
```python
    @staticmethod
    def bert_base2large_aki():
        ckpt_path = f"{root}/ckpt/bert_base.ckpt"
        save_path = f"{root}/output/bert_base2large_aki.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "bert_base", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("bert_large"), "AKI", save_path)
            return new_model

    @staticmethod
    def bert_base2large_fpi():
        ckpt_path = f"{root}/ckpt/bert_base.ckpt"
        save_path = f"{root}/output/bert_base2large_fpi.pth"
        if not os.path.exists(save_path):
            model = ld_py(ckpt_path, "bert_base", load_gitee_ckpt=False, specify_prefix=None)
            new_model = enlarge_py(model, config_py("bert_large"), "FPI", save_path)
            return new_model
```
修改代码MindsporeEnlarger类中的bert_base2large_aki方法和bert_base2large_fpi方法修改如下（对应文件51-67行）：
```python
    @staticmethod
    def bert_base2large_aki():
        ckpt_path = f"{root}/ckpt/bert_base.ckpt"
        save_path = f"{root}/output/bert_base2large_aki.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, kind="bert_base", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("bert_large"), "AKI", save_path)
            return new_model

    @staticmethod
    def bert_base2large_fpi():
        ckpt_path = f"{root}/ckpt/bert_base.ckpt"
        save_path = f"{root}/output/bert_base2large_fpi.ckpt"
        if not os.path.exists(save_path):
            pre_train_model = ld_ms(path=ckpt_path, kind="bert_base", load_gitee_ckpt=False)
            new_model = enlarge_ms(pre_train_model, config_ms("bert_large"), "FPI", save_path)
            return new_model
```

后依次运行

运行BERT模型，需要运行random_init_ckpt_on_gpt.py

依次将```if __name__=='__main__':```内的内容改为
```python
    path = "./ckpt/gpt_base"
    random_init_ckpt(path, "gpt_base")
```
```python
    path = "./ckpt/gpt_medium"
    random_init_ckpt(path, "gpt_medium")
```
```python
    path = "./ckpt/gpt_large"
    random_init_ckpt(path, "gpt_large")
```
```python
    path = "./ckpt/gpt_xl"
    random_init_ckpt(path, "gpt_xl")
```
后依次运行

对BERT模型进行推理验证
```shell
# 运行推理验证示例
# 验证fpi方法用BERT small初始化BERT base
python eval.py --task_type=small2base --method=fpi
# 验证fpi方法用BERT base初始化BERT large
python eval.py --task_type=base2large --method=fpi
# 验证aki方法用BERT small初始化BERT base
python eval.py --task_type=small2base --method=aki
# 验证aki方法用BERT base初始化BERT large
python eval.py --task_type=base2large --method=aki
```
对GPT模型进行推理验证
```shell
# 运行推理验证示例
# 验证fpi方法用GPT base初始化GPT medium
python eval_on_gpt.py --task_type=base2medium --method=fpi
# 验证fpi方法用GPT medium初始化GPT large
python eval_on_gpt.py --task_type=medium2large --method=fpi
# 验证fpi方法用GPT large初始化GPT xl
python eval_on_gpt.py --task_type=large2xl --method=fpi
# 验证aki方法用GPT base初始化GPT medium
python eval_on_gpt.py --task_type=base2medium --method=aki
# 验证aki方法用GPT medium初始化GPT large
python eval_on_gpt.py --task_type=medium2large --method=aki
# 验证aki方法用GPT large初始化GPT xl
python eval_on_gpt.py --task_type=large2xl --method=aki
```
或者运行scripts文件夹下的sh脚本文件。
```shell
# 验证fpi方法用BERT small初始化BERT base
scripts/validate.sh small2base fpi
# 验证fpi方法用BERT base初始化BERT large
scripts/validate.sh base2large fpi
# 验证aki方法用BERT small初始化BERT base
scripts/validate.sh small2base aki
# 验证aki方法用BERT base初始化BERT large
scripts/validate.sh base2large aki
```
```shell
# 验证fpi方法用GPT base初始化GPT medium
scripts/validate_on_gpt.sh base2medium fpi
# 验证fpi方法用GPT medium初始化GPT large
scripts/validate_on_gpt.sh medium2large fpi
# 验证fpi方法用GPT large初始化GPT xl
scripts/validate_on_gpt.sh large2xl fpi
# 验证aki方法用GPT base初始化GPT medium
scripts/validate_on_gpt.sh base2medium aki
# 验证aki方法用GPT medium初始化GPT large
scripts/validate_on_gpt.sh medium2large aki
# 验证aki方法用GPT large初始化GPT xl
scripts/validate_on_gpt.sh large2xl aki
```

如果单独使用扩展功能，则对Bert扩展。可以指定CPU或Ascend。需将预训练ckpt文件放在ckpt目录下。
```shell
# 使用CPU初始化
# 验证fpi方法用BERT small初始化BERT base
python enlarger_bert_mindspore.py --task_type=small2base --method=fpi --device_target="CPU"
# 验证fpi方法用BERT base初始化BERT large
python enlarger_bert_mindspore.py --task_type=base2large --method=fpi --device_target="CPU"
# 验证aki方法用BERT small初始化BERT base
python enlarger_bert_mindspore.py --task_type=small2base --method=aki --device_target="CPU"
# 验证aki方法用BERT base初始化BERT large
python enlarger_bert_mindspore.py --task_type=base2large --method=aki --device_target="CPU"
# 使用Ascend初始化
# 验证fpi方法用BERT small初始化BERT base
python enlarger_bert_mindspore.py --task_type=small2base --method=fpi --device_target="Ascend"
# 验证fpi方法用BERT base初始化BERT large
python enlarger_bert_mindspore.py --task_type=base2large --method=fpi --device_target="Ascend"
# 验证aki方法用BERT small初始化BERT base
python enlarger_bert_mindspore.py --task_type=small2base --method=aki --device_target="Ascend"
# 验证aki方法用BERT base初始化BERT large
python enlarger_bert_mindspore.py --task_type=base2large --method=aki --device_target="Ascend"
```
如果单独使用扩展功能，则对GPT扩展。可以指定CPU或Ascend。
```shell
# 使用CPU初始化
# 验证fpi方法用GPT base初始化GPT medium
python enlarger_gpt_mindspore.py --task_type=base2medium --method=fpi --device_target="CPU"
# 验证fpi方法用GPT medium初始化GPT large
python enlarger_gpt_mindspore.py --task_type=medium2large --method=fpi --device_target="CPU"
# 验证fpi方法用GPT large初始化GPT xl
python enlarger_gpt_mindspore.py --task_type=large2xl --method=fpi --device_target="CPU"
# 验证aki方法用GPT base初始化GPT medium
python enlarger_gpt_mindspore.py --task_type=base2medium --method=aki --device_target="CPU"
# 验证aki方法用GPT medium初始化GPT large
python enlarger_gpt_mindspore.py --task_type=medium2large --method=aki --device_target="CPU"
# 验证aki方法用GPT large初始化GPT xl
python enlarger_gpt_mindspore.py --task_type=large2xl --method=aki --device_target="CPU"
# 使用Ascend初始化
# 验证fpi方法用GPT base初始化GPT medium
python enlarger_gpt_mindspore.py --task_type=base2medium --method=fpi --device_target="Ascend"
# 验证fpi方法用GPT medium初始化GPT large
python enlarger_gpt_mindspore.py --task_type=medium2large --method=fpi --device_target="Ascend"
# 验证fpi方法用GPT large初始化GPT xl
python enlarger_gpt_mindspore.py --task_type=large2xl --method=fpi --device_target="Ascend"
# 验证aki方法用GPT base初始化GPT medium
python enlarger_gpt_mindspore.py --task_type=base2medium --method=aki --device_target="Ascend"
# 验证aki方法用GPT medium初始化GPT large
python enlarger_gpt_mindspore.py --task_type=medium2large --method=aki --device_target="Ascend"
# 验证aki方法用GPT large初始化GPT xl
python enlarger_gpt_mindspore.py --task_type=large2xl --method=aki --device_target="Ascend"
```

# 脚本说明

## 脚本及样例代码

```bash
├── bert2BERT
    ├── README.md                               # 所有模型相关说明
    ├── ckpt                                    # ckpt文件文件夹
    ├── scripts
        ├── infer_mindspore.sh                  # 在Bert模型上的推理脚本
        ├── validate.sh                         # 在Bert模型上的评估脚本
        ├── infer_mindspore_on_gpt.sh           # 在GPT模型上的推理脚本
        ├── validate_on_gpt.sh                  # 在GPT模型上的评估脚本
    ├── src
        ├── Bert_Model
            ├── mindspore_version
                ├── bert_config.py              # Bert模型参数的设定
                ├── bert_mindspore.py           # mindspore版本的Bert模型
                ├── expand_mindspore.py         # mindspore版本扩展
                ├── load_pretrain_model.py      # 生成不同规模的Bert模型
                ├── loader_mindspore.py         # 加载参数并进行相应的扩展
                ├── model_expander_mindspore.py # 模型扩展
                ├── utils.py                    # 工具
            ├── pytorch_version                 # 
                ├── bert_config.py              # Bert模型参数的设定
                ├── expand.py                   # pytorch版本扩展
                ├── huggingface_bert.py         # pytorch版本Bert模型
                ├── load_pretrain_model.py      # 生成不同规模的Bert模型
                ├── loader.py                   # 加载参数并进行相应的扩展
                ├── model_expander.py           # 模型扩展
                ├── utils.py                    # 工具
        ├── GPT_Model
            ├── mindspore_version
                ├── expand_mindspore.py         # mindspore版本扩展
                ├── GPT_config.py               # GPT模型的参数设定
                ├── gpt_mindspore.py            # mindspore版本GPT模型
                ├── load_into_net_mindspore.py  # 将参数导入模型中
                ├── load_pretrain_model.py      # 生成不同规模的GPT模型
                ├── loader_mindspore.py         # 加载参数并进行相应的扩展
                ├── model_expander_mindspore.py # 模型扩展
                ├── weight_init.py              # 权重初始化
                ├── utils                       # 工具
                    ├── find_utils.py
                    ├── task_utils.py
                    ├── tensor_manipulations.py
            ├── pytorch_version
                ├── expand.py                   # pytorch版本扩展
                ├── find_utils.py               # 工具
                ├── GPT_config.py               # GPT模型的参数设定
                ├── huggingface_gpt.py          # pytorch版本GPT模型
                ├── load_pretrain_model.py      # 生成不同规模的GPT模型
                ├── loader.py                   # 加载参数并进行相应的扩展
                ├── model_expander.py           # 模型扩展
    ├── eval.py                                 # 对Bert模型上的效果评估
    ├── eval_on_gpt.py                          # 对GPT模型上的效果评估
    ├── compare.py                              # 比较Bert模型mindspore版本与pytorch版本的误差
    ├── compare_on_gpt.py                       # 比较GPT模型mindspore版本与pytorch版本的误差
    ├── enlarger.py                             # 对Bert模型的扩充方法
    ├── enlarger_on_gpt.py                      # 对GPT模型的扩充方法
    ├── enlarger_bert_mindspore.py              # 可用于Ascend验证
    ├── enlarger_gpt_mindspore.py               # 可用于Ascend验证
    ├── get_path.py                             # 获得文件路径
    ├── infer.py                                # Bert模型的推理
    ├── infer_on_gpt.py                         # GPT模型的推理
    ├── logger.py                               # 生成日志文件
    ├── random_init_ckpt.py                     # 随机生成Bert模型指定规模的ckpt文件
    ├── random_init_ckpt_on_gpt                 # 随机生成GPT模型指定规模的ckpt文件
    ├── requirements.txt                        # 环境要求
    ├── tokenizer.py                            # 基于Bert对数据集进行Token化
    ├── tokenizer_on_gpt.py                     # 基于GPT对数据集进行Token化
```

## 脚本参数

在src/Bert_Model/mindspore_version(pytorch_version)/bert_config中可以配置模型参数

- 配置MindSpore版Bert_small

  ```python
        BertConfig(seq_length=128,
                  vocab_size=21128,
                  hidden_size=512,
                  num_hidden_layers=4,
                  num_attention_heads=8,
                  size_per_head=512 // 8,
                  intermediate_size=2048,
                  hidden_act="gelu",
                  hidden_dropout_prob=0.1,
                  attention_probs_dropout_prob=0.1,
                  max_position_embeddings=512,
                  type_vocab_size=2,
                  initializer_range=0.02,
                  use_relative_positions=False,
                  dtype=mstype.float32,
                  compute_type=mstype.float32)
  ```

- 配置Pytorch版Bert_small

  ```python
        BertConfig(seq_length=128,
                  vocab_size=21128,
                  hidden_size=512,
                  num_hidden_layers=4,
                  num_attention_heads=8,
                  size_per_head=512 // 8,
                  intermediate_size=2048,
                  hidden_act="gelu",
                  hidden_dropout_prob=0.1,
                  attention_probs_dropout_prob=0.1,
                  max_position_embeddings=512,
                  type_vocab_size=2,
                  initializer_range=0.02,
                  use_relative_positions=False,
                  dtype=torch.float32,
                  compute_type=torch.float32)
  ```

bert2Bert从小模型到大模型的扩展是灵活的。用户可以根据需求自行设定BertConfig，然后生成Bert模型。在测试时，选用了官方指定的
small、base
和large配置，它们定义在load_pretrain_model.py中。

在src/GPT_Model/mindspore_version(pytorch_version)/GPT_config中可以配置模型参数

- 配置MindSpore版gpt_base

  ```python
        GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embed=768,
                          n_layer=12,
                          n_head=12,
                          intermediate_size=3072,
                          hidden_act="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=mstype.float32,
                          compute_type=mstype.float16)
  ```

- 配置Pytorch版GPT_base

  ```python
        GPT2Config(batch_size=512,
                          seq_length=1024,
                          vocab_size=50257,
                          n_embd=768,
                          n_layer=12,
                          n_head=12,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)
  ```

bert2Bert从小模型到大模型的扩展是灵活的。用户可以根据需求自行设定GPT2Config，然后生成GPT模型。在测试时，选用了官方指定的
base、medium、large和xl配置，它们定义在load_pretrain_model.py中。

## 评估过程

### 评估



在代码实现时，我们将MindSpore库中mindspore.load_param_into_net函数中的下面代码段删除。该代码段会导致强行匹配不同层的参数 ，以填补缺漏层参数的效果。本任务不需要该功能。我们在本地重写了load_param_into_net函数，而没有使用mindspore库中的该函数。需注意，功能有所差异。


```python
if param_not_load and not strict_load:
    _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load)
```
BERT
  ```bash
  python eval.py --task_type=small2base --method=fpi
  ```
  OR
  ```bash
  bash scripts/validate.sh small2base fpi
  ```
GPT
  ```bash
  python eval_on_gpt.py --task_type=base2medium --method=fpi
  ```
  OR
  ```bash
  bash scripts/validate_on_gpt.sh base2medium fpi
  ```

  上述python命令将在后台运行，您可以通过log/info.log文件查看结果。测试数据集的准确性如下：

  ```bash
  # 模型参数误差
  parameter difference : 0 
  # embedding表示误差
  embedding difference : 0.00023662493913434446
  # 扩展时间花费
  Pytorch : 2.0802080631256104s MindSpore : 113.9900484085083s
  ```

  注：扩展后，程序自动将ckpt或pth文件放到outputs文件夹下。下次调用对比方法时，不再进行重复扩展。


## 推理过程

### 推理

获取预训练或者扩展后的模型后，将其保存在本地文件夹下。调用infer.py，基于SST-2给出的vacab词表进行embedding表示。
需要写明bert的类型，便于导入参数。推理代码只支持MindSpore版，可以指定运行设备为CPU或者Ascend。


BERT模型
1）./output/bert_small2base_fpi.ckpt文件

CPU运行
```shell
python infer.py --sentence="I like it" --ms_bert_path="./output/bert_small2base_fpi.ckpt" --kind="bert_base" --device_target="CPU"
```
OR
```shell
bash scripts/infer_mindspore.sh "I like it" "./output/bert_small2base_fpi.ckpt" "bert_base" "CPU"
```
Ascend运行
```shell
python infer.py --sentence="I like it" --ms_bert_path="./output/bert_small2base_fpi.ckpt" --kind="bert_base" --device_target="Ascend"
```
OR
```shell
bash scripts/infer_mindspore.sh "I like it" "./output/bert_small2base_fpi.ckpt" "bert_base" "Ascend"
```

2）./output/bert_small2base_aki.ckpt文件

CPU运行
```shell
python infer.py --sentence="I like it" --ms_bert_path="./output/bert_small2base_aki.ckpt" --kind="bert_base" --device_target="CPU"
```
OR
```shell
bash scripts/infer_mindspore.sh "I like it" "./output/bert_small2base_aki.ckpt" "bert_base" "CPU"
```
Ascend运行
```shell
python infer.py --sentence="I like it" --ms_bert_path="./output/bert_small2base_aki.ckpt" --kind="bert_base" --device_target="Ascend"
```
OR
```shell
bash scripts/infer_mindspore.sh "I like it" "./output/bert_small2base_aki.ckpt" "bert_base" "Ascend"
```

3）./output/bert_base2large_fpi.ckpt文件

CPU运行
```shell
python infer.py --sentence="I like it" --ms_bert_path="./output/bert_base2large_fpi.ckpt" --kind="bert_large" --device_target="CPU"
```
OR
```shell
bash scripts/infer_mindspore.sh "I like it" "./output/bert_base2large_fpi.ckpt" "bert_large" "CPU"
```
Ascend运行
```shell
python infer.py --sentence="I like it" --ms_bert_path="./output/bert_base2large_fpi.ckpt" --kind="bert_large" --device_target="Ascend"
```
OR
```shell
bash scripts/infer_mindspore.sh "I like it" "./output/bert_base2large_fpi.ckpt" "bert_large" "Ascend"
```

4）./output/bert_base2large_aki.ckpt文件

CPU运行
```shell
python infer.py --sentence="I like it" --ms_bert_path="./output/bert_base2large_aki.ckpt" --kind="bert_large" --device_target="CPU"
```
OR
```shell
bash scripts/infer_mindspore.sh "I like it" "./output/bert_base2large_aki.ckpt" "bert_large" "CPU"
```
Ascend运行
```shell
python infer.py --sentence="I like it" --ms_bert_path="./output/bert_base2large_aki.ckpt" --kind="bert_large" --device_target="Ascend"
```
OR
```shell
bash scripts/infer_mindspore.sh "I like it" "./output/bert_base2large_aki.ckpt" "bert_large" "Ascend"
```

GPT模型

1）./output/gpt_base2medium_fpi.ckpt文件

CPU运行
```shell
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_base2medium_fpi.ckpt" --kind="gpt_medium" --device_target="CPU" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_base2medium_fpi.ckpt" "gpt_medium" "CPU"
```
Ascend运行
```bash
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_base2medium_fpi.ckpt" --kind="gpt_medium" --device_target="Ascend" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_base2medium_fpi.ckpt" "gpt_medium" "Ascend"
```

2）./output/gpt_base2medium_aki.ckpt文件

CPU运行
```shell
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_base2medium_aki.ckpt" --kind="gpt_medium" --device_target="CPU" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_base2medium_aki.ckpt" "gpt_medium" "CPU"
```
Ascend运行
```bash
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_base2medium_aki.ckpt" --kind="gpt_medium" --device_target="Ascend" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_base2medium_aki.ckpt" "gpt_medium" "Ascend"
```

3）./output/gpt_medium2large_fpi.ckpt文件

CPU运行
```shell
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_medium2large_fpi.ckpt" --kind="gpt_large" --device_target="CPU" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_medium2large_fpi.ckpt" "gpt_large" "CPU"
```
Ascend运行
```bash
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_medium2large_fpi.ckpt" --kind="gpt_large" --device_target="Ascend" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_medium2large_fpi.ckpt" "gpt_large" "Ascend"
```

4）./output/gpt_medium2large_aki.ckpt文件

CPU运行
```shell
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_medium2large_aki.ckpt" --kind="gpt_large" --device_target="CPU" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_medium2large_aki.ckpt" "gpt_large" "CPU"
```
Ascend运行
```bash
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_medium2large_aki.ckpt" --kind="gpt_large" --device_target="Ascend" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_medium2large_aki.ckpt" "gpt_large" "Ascend"
```

5）./output/gpt_large2xl_fpi.ckpt文件

CPU运行
```shell
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_large2xl_fpi.ckpt" --kind="gpt_xl" --device_target="CPU" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_large2xl_fpi.ckpt" "gpt_xl" "CPU"
```
Ascend运行
```bash
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_large2xl_fpi.ckpt" --kind="gpt_xl" --device_target="Ascend" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_large2xl_fpi.ckpt" "gpt_xl" "Ascend"
```

6）./output/gpt_large2xl_aki.ckpt文件

CPU运行
```shell
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_large2xl_aki.ckpt" --kind="gpt_xl" --device_target="CPU" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_large2xl_aki.ckpt" "gpt_xl" "CPU"
```
Ascend运行
```bash
python infer_on_gpt.py --sentence="I like it" --ms_gpt_path="./output/gpt_large2xl_aki.ckpt" --kind="gpt_xl" --device_target="Ascend" 
```
OR
```bash
bash scripts/infer_mindspore_on_gpt.sh "I like it" "./output/gpt_large2xl_aki.ckpt" "gpt_xl" "Ascend"
```
# 模型描述

## 性能

### 评估性能

#### 在SST-2数据集上，输入语句"My dog is cute." 进行比较。

| 参数(Bert)                   | small2base_aki  | base2large_aki  | 
|----------------------------|-----------------|-----------------|
| 资源                         | cpu             | cpu             |
| 参数误差                       | 0               | 0               |
| embedding误差                | 0.1%            | 0.4%            |
| embedding维度                | 768             | 1024            |
| 扩展用时(MindSpore/Pytorch)(s) | 113.9900/2.0802 | 174.6869/5.7470 |
| 推理模型                       | 390MB           | 1.02GB          |

| 参数(GPT)                    | base2medium_aki  | medium2large_aki | large2xl_aki      | 
|----------------------------|------------------|------------------|-------------------|
| 资源                         | cpu              | cpu              | cpu               |
| 参数误差                       | 0                | 0                | 0                 |
| embedding误差                | 0.3%             | 0.2%             | 0.5%              |
| embedding维度                | 768              | 1024             | 1280              |
| 扩展用时(MindSpore/Pytorch)(s) | 337.1350/13.0278 | 793.2793/29.3106 | 1938.2301/49.5891 |
| 推理模型                       | 1.13GB           | 2.17GB           | 3.89GB            |

## 使用流程

### 扩展

如果您需要使用bert2Bert扩展模型。下面是操作步骤示例：

```python
# 扩展bert
origin_bert = ...# 填入您想扩展的小规模bert，其参数是预训练完毕的。
target_bert_config = BertConfig(vocab_size=*, # 大型bert的参数设定
            hidden_size=*,
            num_hidden_layers=*,
            num_attention_heads=*,
            size_per_head=*,
            intermediate_size=*,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0)
method = "fpi" # or "aki"
expanded_model = expand_bert(origin_bert, target_bert_config, method)
```

```python
# 扩展gpt
origin_gpt = ...# 填入您想扩展的小规模gpt，其参数是预训练完毕的。
target_gpt_config = GPT2Config(batch_size=512, # 大型GPT的参数设定
                          seq_length=1024,
                          vocab_size=*,
                          n_embd=*,
                          n_layer=*,
                          n_head=*,
                          n_inner=3072,
                          activation_function="gelu",
                          hidden_dropout=0.1,
                          attention_dropout=0.1,
                          n_positions=1024,
                          initializer_range=0.02,
                          input_mask_from_dataset=True,
                          summary_first_dropout=0.1,
                          dtype=torch.float32,
                          compute_type=torch.float16)
method = "fpi" # or "aki"
expanded_model = expand_GPT(origin_gpt, target_gpt_config, method)
```

### 推理[]()
```python
_infer_mindspore_bert(sentence, ckpt_path, bert_kind, device_target)
```
```python
_infer_mindspore_gpt(sentence, ckpt_path, gpt_kind, device_target)
```

# 随机情况说明

在eval.py与eval_on_gpt中，我们设置了PytorchEnlarger和MindsporeEnlarger类内方法的种子。以便两个版本展示出相同的结果。

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
