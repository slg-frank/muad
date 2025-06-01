# MUAD Project

Project structure:

```
MUAD/
├── configs/                  # configuration files
│   └── params.json           # model parameter configuration
├── data/                     # data loading and processing
│   ├── __init__.py
│   ├── dataset.py            # dataset class definitions
│   └── utils.py              # data processing utilities
├── models/                   # model definitions
│   ├── __init__.py
│   ├── encoders.py           # multimodal encoders
│   ├── graph_model.py        # graph neural network components
│   ├── layers.py             # basic network layers
│   └── main_model.py         # main model integration
├── training/                 # training related
│   ├── __init__.py
│   └── base_model.py         # base training framework
├── active_learning/          # active learning module
│   ├── __init__.py
│   └── strategies.py         # active learning strategies
├── utils/                    # utility functions
│   ├── __init__.py
│   ├── general_utils.py      # general utilities
│   └── logging_utils.py      # logging utilities
├── main.py                   # main entry script
└── README.md                 # project documentation
```

The instructions for running MUAD are as follows.

```bash
python main.py
```

