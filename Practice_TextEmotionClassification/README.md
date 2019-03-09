# 科赛：文本情感分类模型搭建 | 练习赛

[比赛链接](https://www.kesci.com/home/competition/5c77ab9c1ce0af002b55af86/leaderboard)

### 处理数据集

1：读取数据并分离label

2：得到去重后的语料库，并制作字典映射语料库

3：文本向量化

4：设置文本填充，统一shape

### 模型设置

1：使用keras中的Embedding LSTM Dense 搭建baseline模型。（未加深模型，添加dropout 、L2等）

2：编译模型

3：未设置Callback类（后期会添上）

4：训练模型

5：未可视化训练结果（acc、loss 与epoch关系）

### 预测结果

1：定义单条数据预测函数

2：循环测试集进行预测

### 提交结果

保存本地文件csv到科赛比赛链接中提交结果



