This is a Keras implementation of "[A 12-Lead ECG Arrhythmia Classification Method Based on 1D Densely Connected CNN](https://link.springer.com/book/10.1007%2F978-3-030-33327-0)" paper, which is also a 3rd prize solution of "[The First China ECG Intelligent Competition](https://www.dcjingsai.com/v2/cmptDetail.html?id=277)".

### Contents

- final_train folder:  
contains the training code.
```bash
python attention_12leads_10nets.py
python attention_12leads_1nets.py
python densenet_12leads_10nets.py
```

- final_run_semi folder:  
contains the predicting code of semi-final.

- final_run_final folder:  
contains the predicting code of final.


The codes contains the following components:

* Python scripts:
   -- challenge.py (necessary) -  add your codes to classify normal and diseases.For ease of evaluation, you should pay attention to the following points:
   1.You need to write the results into "answers.csv",and save it in the current folder
   2.You need to write your test data path with the argparse parameter
   In short, challenge.py is your test code to make predictions or inferences. Please refer to this demo file for details.


* BASH scripts:
   -- run.sh (necessary) - a script calls "challenge.py" to generate "answers.csv", you can modify the --test_path parameter in this file
     
	 
* CSV files:
   -- answers.csv (necessary) - a text file containing the prediction results.

* README.txt - this file

* Other files:
     These files support to run the bash file and the challenge.py, such as your codes to run the model, and the model file, etc.

We verify that your code is working as you intended, by running "run.sh" on the test set, then comparing the results with references.

### Details
1、See the [The First China ECG Intelligent Competition](http://mdi.ids.tsinghua.edu.cn/#/) webpage for more details, including description and results of the competition.   
2、See the [First International Workshop, MLMECH 2019](https://link.springer.com/book/10.1007%2F978-3-030-33327-0), Held in Conjunction with MICCAI 2019 webpage for more solutions, including papers and solutions of the "The First China ECG Intelligent Competition".  

