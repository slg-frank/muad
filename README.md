# MUAD Project

Project structure：

```
MUAD/
├── configs/                  # 配置文件
│   └── params.json           # 模型参数配置
├── data/                     # 数据加载和处理
│   ├── __init__.py
│   ├── dataset.py            # 数据集类定义
│   └── utils.py              # 数据处理工具
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── encoders.py           # 多模态编码器
│   ├── graph_model.py        # 图神经网络组件
│   ├── layers.py             # 基础网络层
│   └── main_model.py         # 主模型集成
├── training/                 # 训练相关
│   ├── __init__.py
│   └── base_model.py         # 基础训练框架
├── active_learning/          # 主动学习模块
│   ├── __init__.py
│   └── strategies.py         # 主动学习策略
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── general_utils.py      # 通用工具
│   └── logging_utils.py      # 日志工具
├── main.py                   # 主入口程序
└── README.md                 # 项目文档
```

The instructions for running model MUAD are as follows.

```python
python main.py
```

