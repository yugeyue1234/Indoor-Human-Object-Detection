#  RIS-based human target detection
 As shown in Figure 1, a schematic diagram of RIS-based human target detection is shown. A dataset was obtained by using RIS to control WiFi signals to scan moving human bodies indoors. Finally, the machine learning algorithm was used to process the dataset to get the room and the localization accuracy.
 <div align=center>
<img src="fig1.png"> 
</div>

## 1. Get datasets：Note: the previous part represents the data type and the numeric part represents. How many monitoring points were usedFor example: ASBTS (2) ： ASBTS represents the data type; (2) represents that two monitoring points were used.
Datasets in folder: Target detection dataset

## 2. Requirements
- matlab
- matplotlib
- numpy 
## 3. How to use it
Download the dataset and place it in the same folder as the code
### 3.1 Run with ConvNet and SVM classification model
```python 
a; first run gate.py - import the dataset into it and use this code to partition the dataset
b; then run SVM.py and train_model3.py, respectively, to find the detection accuracy.
```
### Note: CNN confusion matrix is given together with the detection accuracy in train_model3.py.Accuracy comparison plots with draw2.py.

## 4. Dataset size description
Three commonly utilized datasets for machine learning provide the data for modulation identification: RML2016.10a_dict.pkl, 2016.04C.multisnr.pkl, and RML2016.10b.dat. The samples in the dataset are superimposed by the attack interference to generate new datasets RML2016.10a_int.pkl, RML2016.10b_int.pkl, and RML2016.04c_int.pkl. The addition of INR resulted in an 11-fold increase in the size of the new dataset, as shown in the following table:
|Dataset name|Sample dimension|Datasize|SNR range(dB)|INR range(dB)|Modulation schemes|
|:--:|:--:|:--:|:--:|:--:|:--:|
|RML2016.10a_dict|2×128|220000|-20:2:18|——|11|
|RML2016.10a_int|2×128|2420000|-20:2:18|-10:2:10|11|
|RML2016.10b|2×128|1200000|-20:2:18|——|10|
|RML2016.10c_int|2×128|13200000|-20:2:18|-10:2:10|10
|2016.04c.multisnr|2×128|162060|-20:2:18|——|11|
|RML2016.04c_int|2×128|1782660|-20:2:18|-10:2:10|11|
