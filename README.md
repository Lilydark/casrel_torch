# casrel_torch

基于torch的casrel关系抽取算法
可在config.py中调整设置，如果需要训练其他数据，仿照dataset/baidu中的数据格式修改

与实体识别+关系识别的分步抽取方法相比，Casrel作为联合抽取方案，适合处理实体数量较少但关系较多且易重叠的关系抽取任务。
