# RQ3



## Run different sampling methods

### Use a trained model
1.Use the results we have reserved, or save the results you obtained for rq1 and rq2 to the same directory.

2.Using the command line: run the following command:

```
python GA_n.py
python lower_power.py
python uni_sampling.py
```


### Results presentation

1.Ensure that you have sufficient output folders and that the filenames correspond correctly.

2.Using the command line: run the following command:

```
python show_result.py
```
You can obtain results in the terminal, replace ***folder_name*** to obtain the results of different sampling methods.

### Deploy NDP in the real-world system
1.Ensure that the output file paths for rq1 and rq2 are correct.

2.Divide the sampling points according to the range of configuration options.

3.Edit the *GA_n.py* file to ensure that the performance test results you want are transferred to the corresponding position in the *perf_list*.

4.Run your customized *GA_n.py* file until suspicious options are detected or the preset threshold is exceeded.

**Note**: Performance test results come in many forms, please make sure to select the analysis preference according to the actual effect. More details can be obtained in our paper.

