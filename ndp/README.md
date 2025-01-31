# Deploy NDP in the real-world system

## Installation Requirements

1.Download all the files into the same folder/clone the repository.

2.Make sure you are using Python version between 3.7 and 3.9. Otherwise, you may encounter unknown errors.

3.Using the command line: cd to the folder with the codes, and install all the required packages by running:
```
pip install -r requirements.txt
```

## Data preparation

1.Replace the data(includes training and test data) with the system you want to deploy, packing the training data into datasets format and put them into the *data_pp* and *data_pr* folders respectively.


2.Download [pre-trained model](https://huggingface.co/FacebookAI/roberta-base) and put them into the *data_pp* and *data_pr* folders respectively.


## Run NDP

1.Edit and run *roberta_PP_KF_gpu.py* to obtain the prediction results of CPBugs types.

2.Edit and run *roberta_pr_kde_gpu.py* to get the results of Option-CPBugs relevance estimation, and divide options with different densities to different files.

3.(Optional) Select options related to the buffer and process them using high density methods.

4.Divide the sampling points according to the range of configuration options.

5.Edit the *GA_n.py* file. You can put the data that has been tested into the ***perf_list*** for quick sampling. And you can also replace the ***perf_list*** to conduct immediate tests on the real-world software system according to the value of the currently selected test option to obtain performance data.

6.Run your customized *GA_n.py* file until suspicious options are detected or the preset threshold is exceeded.

**Note**: Performance test results come in many forms, please make sure to select the analysis preference according to the actual effect. More details can be obtained in our paper. You can deploy different versions of real-world systems through docker. We provide some commonly used dockerfile and deployment steps which is available at：https://github.com/ideas-labo/ndp/tree/main/dockerfiles.
