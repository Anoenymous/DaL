# Predicting Software Performance with Divide-and-Learn
This repository contains the key codes, full data used, and the suppplementary tables in the paper **_'Predicting Software Performance with Divide-and-Learn'_**.

# Documents

- **DaL_main.py**: 
the *main program* for using DaL, which automatically reads data from csv files, trains and evaluates DaL as well as the other local models as specified in the paper.

- **mlp_plain_model.py**:
contains functions to construct and train plain DNN. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf).
    
- **mlp_sparse_model.py**:
contains functions to construct and build DNN with L1 regularization. This is also used by [DeepPerf](https://github.com/DeepPerf/DeepPerf).

- **utils**

    └─ **general.py**:
    contains utility functions to build DNN and other ML models.
    
    └─ **hyperparameter_tuning.py**:
    contains function that efficiently tunes hyperparameters of DNN.
    

- **Data**:
performance datasets of 8 subject systems as specified in the paper.

- **Tables**:
supplementary tables for the paper.

# Prerequisites and Installation
1. Download all the files into the same folder.

2. The codes are tested with *Python 3.6 - 3.7* and *Tensorflow 1.x*, other versions might cause errors.

3. Run *DaL_main.py* and install all missing packages according to runtime messages.


# Run DaL

- **Command line**: cd to the folder with the codes, and simply run

        python DaL_main.py
        
- **Python IDE (e.g. Pycharm)**: Open the *DaL_main.py* file on the IDE, and simply click 'Run'.



# Change Experiment Settings
The main program is fully automated with a default experiment setting, which is an example of evaluating DaL.

To run more complicated experiments, alter the codes following the comments in *DaL_main.py*.

#### To switch between subject systems
    Comment and Uncomment the lines 33-40 following the comments in DaL_main.py.

    E.g., to run DaL with Apache, uncomment line 33 'subject_system = 'Apache_AllNumeric'' and comment out the other lines.


#### To save the experiment results
    Set 'save_file = True' at line 21.


#### To change the number of experiments
    Change 'N_experiments' at line 27, where each element corresponds a sample size. 

    For example, to simply run the first sample size with 30 repeated runs, set 'N_experiments = [30, 0, 0, 0, 0]'.

#### To change the sample sizes of a particular system
    Edit lines 55-71.

    For example, to run Apache with sample sizes 10, 20, 30, 40 and 50: set line 55 with 'sample_sizes = [10, 20, 30, 40, 50]'.


#### To compare DaL with DeepPerf
    1. Set line 20 with 'test_mode = False'.

    2. Set line 23 with 'enable_deepperf = True'.


#### To compare DaL with other ML models (RF, DT, LR, SVR, KRR, kNN) and DaL framework with these models (DaL_RF, DaL_DT, DaL_LR, DaL_SVR, DaL_KRR, DaL_kNN)
    1. Set line 20 with 'test_mode = False'.

    2. Set line 22 with 'enable_baseline_models = True'.


#### To run DaL with different depth d
    Run DaL with d=2: set line 25 with 'depths = [2]'.

    Run DaL with d=3 and d=4: set line 25 with 'depths = [3, 4]'.


# State-of-the-art Performance Prediction Models
Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with DaL in the paper. 

- [DeepPerf](https://github.com/DeepPerf/DeepPerf)

    A deep neural network performance model with L1 regularization and efficient hyperparameter tuning.

- [DECART](https://github.com/jmguo/DECART)

    CART with data-efficient sampling method.

- [SPLConqueror](https://github.com/se-sic/SPLConqueror)

    Linear regression with optimal binary and numerical sampling method and stepwise feature seleaction.

- [Perf-AL](https://github.com/GANPerf/GANPerf)

    Novel GAN based performance model with a generator to predict performance and a discriminator to distinguish the actual and predicted labels.
    


Note that *DaL_main.py* only contains DeepPerf because it is formulated in the most similar way to DaL, while the others are developed under different programming languages or have differnt ways of usage. 

Therefore, to compare DaL other SOTA models, please refer to their original pages (You might have to modify or replicate their codes to ensure the compared models share the same set of training and testing samples).
