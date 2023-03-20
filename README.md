#  RIS-based human target detection
 As shown in Figure 1, a schematic diagram of RIS-based human target detection is shown. A dataset was obtained by using RIS to control WiFi signals to scan moving human bodies indoors. Finally, the machine learning algorithm was used to process the dataset to get the room and the localization accuracy.
 <div align=center>
<img src="fig1.png"> 
</div>

## 1. Get datasets
Dataset for RIS-based human target detection：

 [link] (https://www.deepsig.ai/datasets)

## 2. Requirements
- matlab
- matplotlib
- numpy 
## 3. How to use it
Download the dataset and place it in the same folder as the code
### 3.1 Run with ConvNet classification model
```python 
python Gen_interference_dataset.py
```
### 3.2 Run with other classification models
1. Use your classification model to train the data and get the classification model parameters.

2. With your model and parameters, replace the following in the Gen_interference_dataset.py：
```python
   model = ConvNet128(len(all_data[-1])).to(device=device)  
   model.load_state_dict(torch.load('./data/saved_Convnetmodel(rml2016c).pth', map_location=device))  
   model.eval()
```
## 4. Comparison of old and new datasets of RML2016
Three commonly utilized datasets for machine learning provide the data for modulation identification: RML2016.10a_dict.pkl, 2016.04C.multisnr.pkl, and RML2016.10b.dat. The samples in the dataset are superimposed by the attack interference to generate new datasets RML2016.10a_int.pkl, RML2016.10b_int.pkl, and RML2016.04c_int.pkl. The addition of INR resulted in an 11-fold increase in the size of the new dataset, as shown in the following table:
|Dataset name|Sample dimension|Datasize|SNR range(dB)|INR range(dB)|Modulation schemes|
|:--:|:--:|:--:|:--:|:--:|:--:|
|RML2016.10a_dict|2×128|220000|-20:2:18|——|11|
|RML2016.10a_int|2×128|2420000|-20:2:18|-10:2:10|11|
|RML2016.10b|2×128|1200000|-20:2:18|——|10|
|RML2016.10c_int|2×128|13200000|-20:2:18|-10:2:10|10
|2016.04c.multisnr|2×128|162060|-20:2:18|——|11|
|RML2016.04c_int|2×128|1782660|-20:2:18|-10:2:10|11|
