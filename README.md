This repository contains the key codes and full data used in the paper **_'Predicting Software Performance with Divide-and-Learn'_**.

# Documents

- **DaL_main.py**: 
The **main program** for using DaL, which reads data from csv files, trains and evaluates DaL as well as the other local models as specified in the paper.

- **mlp_plain_model.py**:
Contains functions to construct and train plain DNN. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf).
    
- **mlp_sparse_model.py**:
Contains functions to construct and build DNN with L1 regularization. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf).

- **utils**

    └─ **general.py**:
    Contains utility functions to build DNN and other ML models.
    
    └─ **hyperparameter_tuning.py**:
    Contains function that efficiently tunes hyperparameters of DNN.
    

- **Data**:
Performance datasets of 8 subject systems as specified in the paper.

- **Tables**:
Supplementary tables for the paper.

# State-of-the-art Models
Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with DaL in the paper:

- [DeepPerf](https://github.com/DeepPerf/DeepPerf)

    A deep neural network performance model with L1 regularization and efficient hyperparameter tuning.

- [DECART](https://github.com/jmguo/DECART)

    CART with data-efficient sampling method.

- [SPLConqueror](https://github.com/se-sic/SPLConqueror)

    Linear regression with optimal binary and numerical sampling method and stepwise feature seleaction.

- [Perf-AL](https://github.com/GANPerf/GANPerf)

    Novel GAN based performance model with a generator to predict performance and a discriminator to distinguish the actual and predicted labels.

# Prerequisites and Installation

# Usage
