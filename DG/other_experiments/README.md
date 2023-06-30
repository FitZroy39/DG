此处包含一些做过的拓展试验，最终上线的模型并没有用到。这里进行简单介绍以方便后续跟进。
实验包括二分类加入observability通道、bev进行多分类、pointpillar提取特征。

## observability通道
仿照论文《MASS: Multi-Attentional Semantic Segmentation of LiDAR Data for Dense Top-View Understanding》为bev添加一个observability通道，具体计算方法如下：

对于每个点云，将其与原点相连，从原点出发，该射线经过的grid计数加一，直到遇到含障碍物的grid终止，最后可形成HxW的特征图，
将该特征图除以最大元素以归一化，与原始的bev图合并。observability效果如下：

![image](other_experiments/observability.png)

计算代码在`dataset/pc_bev.py`中，但效果并没有提升，因此最终没有使用。

## 多分类
目的是借助检测模型的label进行grid多分类。多分类的标签生成可以参考`generate_obstacle_gt_xxx.py`。实验的设定和部分结果记录参考[多分类结果记录](http://jira.fabu.ai/browse/PER-59)

## pointpillar提取特征
除了用bev提取特征之外，还尝试使用pointpillar自动学习点云特征，主要代码在`datasets/pointpillar`和`core`中。最终效果不如bev，因此没有使用。

| -  | IOU | precision | recall | f1-score |
| --- | --- | --- | --- | --- |
| bev | 0.772 | 0.876 | 0.867 | 0.871 |
| point pillar | 0.753 | 0.870 | 0.849 | 0.859 |