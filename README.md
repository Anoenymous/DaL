This repository contains the key codes and full data used in the paper **_'Predicting Software Performance with Divide-and-Learn'_**.

# Documents

- **DaL_main.py**:
The main program for using DaL, which reads data from csv files, trains and evaluates DaL as well as the other local models as specified in the paper.

- **mlp_plain_model.py**:
Contains functions to construct and train plain DNN. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf).
    
- **mlp_sparse_model.py**:
Contains functions to construct and build DNN with L1 regularization. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf).

- **utils**

    └─ **general.py**:
    Contains utility functions to build DNN.
    
    └─ **hyperparameter_tuning.py**:
    Contains function that efficiently tunes hyperparameters of DNN.
    

- **Data**:
Performance datasets of 8 subject systems as specified in the paper.

# Documents
