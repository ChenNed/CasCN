# CasCN
This is a TensorFlow implementation of Recurrent Cascades Convolution for the task of information cascades prediction, And the paper "Information Diffusion Prediction via Recurrent Cascades Convolution" (Accepted by ICDE 2019) will be avilable to download at the internet soon.
# Overview
- `data/` put the download dataset here;
- `model/` contains the implementation of the CasCN;
- `preprocessing/` contains preprocessing code：
    * split the data to train set, validation set and test set (`utils.py`);
    * trainsform the datasets to the format of ".pkl" (`preprocess_graph_signal.py`)
    * (`config.py`) you can configure parameters and filepath in this file
    .
# Datatset
The datasets we used in our paper are Sina Weibo and HEP-PH. For the Sina Weibo dataset, you can download [here](https://github.com/CaoQi92/DeepHawkes) and the HEP-PH dataset is avilable [here](http://snap.stanford.edu/data/cit-HepPh.html).
Also, we provide a pre-processed Weibo dataset (T=3 hours) [here](https://pan.baidu.com/s/1_s3FvbEpj2piWcRqLqpb5A) and the file password is: (`a7xu`)

Steps to run CasCN
----------------------------------- 

1.split the data to train set, validation set and test set. Then trainsform the datasets to the format of ".pkl"
command: 

    cd preprocessing
    python utils.py
    python preprocess_graph_signal.py
 
2.train Model
command:

    cd model
    python run_graph_sequence.py
# Notice
 If you want to do the experiment with citation dataset - "HEP-PE", you should first transform the format of citation dataset as the same as Weibo dataset. (the format of Weibo dataset you can reference [here](https://github.com/CaoQi92/DeepHawkes)). And the version of the Tensorflow we used is 1.0.

