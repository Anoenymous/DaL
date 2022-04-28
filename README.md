This repository contains the key codes and full data used in the paper **_'Predicting Software Performance with Divide-and-Learn'_**.

# Documents

- **DaL_main.py**: 
The *main program* for using DaL, which reads data from csv files, trains and evaluates DaL as well as the other local models as specified in the paper.

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

# Prerequisites and Installation
1. Download all the files into the same folder.

2. The codes are tested with *Python 3.6 - 3.7* and *Tensorflow 1.x*, other versions might cause errors.

3. Run *DaL_main.py* and install all missing packages according to runtime messages.

# Usage
## To run DaL
- **Command line**: cd to the folder with the codes, and run

        python DaL_main.py
        
- **Python IDE (e.g. Pycharm)**: Open the Encoding.py file on the IDE, and click 'Run'.



## To switch between subject systems
Comment and Uncomment the lines 33-40 following the comments in *DaL_main.py*.

E.g., to run DaL with Apache, uncomment line 33 'subject_system = 'Apache_AllNumeric'' and comment the other lines.



## To change experiment settings
Alter the codes between lines 20-30 following the comments in *DaL_main.py*.

E.g.:
- To save the experiment results, set 'save_file = True' at line 21.
- To change the number of experiments, change 'N_experiments' at line 27, where each element corresponds a sample size. 
For example, to simply run the first sample size with 30 repeated runs, set 'N_experiments = [30, 0, 0, 0, 0]'.


## To compare DaL with DeepPerf
1. Set line 20 with 'test_mode = False'

2. Set line 23 with 'enable_deepperf = True'


## To compare DaL with other ML models (RF, DT, LR, SVR, KRR, kNN) and DaL framework with these models (DaL_RF, DaL_DT, DaL_LR, DaL_SVR, DaL_KRR, DaL_kNN)
1. Set line 20 with 'test_mode = False'

2. Set line 22 with 'enable_baseline_models = True'



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
