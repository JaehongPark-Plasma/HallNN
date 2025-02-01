# HallNN: Code and training dataset for "Predicting performance of Hall effect ion source using machine learning"
**Hall** thruster performance prediction with **N**eural **N**etwork ensemble   
Online publised, Advanced Intelligent Systems (2024)  
DOI: https://doi.org/10.1002/aisy.202400555

## HallNN structure
### **Ensemble of 100 neural networks**

![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/Data/intro.png?raw=true)

* **13 inputs (thruster design and operating parameters)**
  * Anode flow rate (sccm)
  * Voltage drop (V)
  * Outer channel radius (mm)
  * Inner channel radius (mm)
  * Discharge channel length (mm)
  * Eight radial magnetic field fitting coefficients
  * (See *Information of training dataset* for details)
* **4 outputs**
  * Thrust & Prediction uncertainty (standard deviation)
  * Discharge current & Prediction uncertainty (standard deviation)


***

## Information of training dataset
### Information of VirtualHallThruster_Dataset.csv and .xlsx  
Contains 4,500 **"virtual"** Hall thruster performance data points, which were generated with numerical simulation (in-house KEPSi-1D code) where Hall thruster design & operating parameters were generated ramdomly.  
Located in Data\Dataset_in_xlsx_and_csv folder.  

Rows from 1 to 13 are **INPUT** parameters   
> 1. Anode flow rate (SCCM), 1 sccm = 0.09763 mg/s relationship was used.
> 2. Voltage drop (V) = Anode voltage - cathode coupling voltage
> 3. Outer channel radius (mm)
> 4. Inner channel radius (mm)
> 5. Discharge channel length (mm)
> 6. to 13. Radial magnetic field fitting coefficients

Rows from 14 and 15 are **THRUST** and **DISCHARGE CURRENT**   
> 14. Thrust (mN)
> 15. Discharge current (A)

Rows from 16 and 17 are **Standard Deviation** of **THRUST** and **DISCHARGE CURRENT** from numerical simulations.
> 14. STD of thrust (mN)
> 15. STD of discharge current (A)

Note) Standard deviations of 'empirical parameter average method' result, which were *not used* in the training process (see the paper).    

### Information of NN_VHT_in.mat and NN_VHT_out.mat
Contains 4,500 **"virtual"** Hall thruster performance data points.   
*NN_VHT_in.mat* is the input mat file (13 parameters) for the MATLAB code.   
*NN_VHT_out.mat* is the output mat file (**thrust** and **discharge current**) for the MATLAB code.   

### Vritual Hall thruster (VHT) dataset overview

Input parameters $\leftrightarrow$ Thrust         |  Input parameters $\leftrightarrow$ Discharge current
:-------------------------:|:-------------------------:  
![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/Data/Input_thrust_VHT.png?raw=true)  |  ![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/Data/Input_Id_VHT.png?raw=true)

***

## Prediction using trained neural networks 
### PREDICTION_with_pretrained_HallNN.m (manuscript version)  
Requirements: Deep Learning Toolbox / Curve Fitting Toolbox  

This code generates HallNN Prediction Results with KHT-40, 200 W-class KAIST Hall thruster (FM) with pre-trained neural networks from the manuscript.
> Anode mass flow rate: 6 to 13 sccm  
> Va, Anode voltage: 250 V  
> Vc, cathode coupling voltage: 30 V  
> R_out, outer channel radius: 20 mm  
> R_in, inner channel radius: 13.5 mm  
> L_ch, channel length: 25 mm  
> B_1-8: 341.2550, 22.7500, 11.8458, 1.9528, 5.5918, 23.2564, 1.4190, 8.7950  

```matlab
% Load manuscript version HallNN
load('Pretrained_HallNN.mat');
NNens = HallNN_best{1};
net_ens_no = HallNN_best{3};
NN_minmax = HallNN_best{9};
```
It uses **Pretrained_HallNN.mat** that contains 100 neural networks information trained with 18,000 datapoints.  
> **Pretrained_HallNN.mat**
> > HallNN_best{1}: 100 neural networks  
> > HallNN_best{2}: informantion of 100 neural networks  
> > HallNN_best{3-8}: M, H1, H2, adversarial rate, early stop, test fraction  
> > HallNN_best{9}: min_max of input and output

Thrust (HallNN)          |  Discharge current (HallNN)  
:-------------------------:|:-------------------------:  
![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/results/HallNN_KHT40_AFR_Thrust_V250.png?raw=true)  |  ![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/results/HallNN_KHT40_AFR_Id_V250.png?raw=true)  
   
### You can also chose to use Virtual Hall Thruster dataset trained HallNN - "ressults/HallNN_VHTver.mat"  
```matlab
% Load VHT-dataset-only-trained version HallNN -> Now it is a test case
load('results/HallNN_VHTver.mat');
NNens = NN_result{1};
net_ens_no = NN_result{3};
NN_minmax = NN_result{9};
```
It uses **results/HallNN_VHTver.mat** that contains 100 neural networks information trained with 4,500 VHT dataset (virtual Hall thruster).  
> **results/HallNN_VHTver.mat**
> > NN_result{1}: 100 neural networks
> > NN_result{2}: informantion of 100 neural networks  
> > NN_result{3-8}: M, H1, H2, adversarial rate, early stop, test fraction  
> > NN_result{9}: min_max of input and output

Therefore, this prediction result represents a test case since the KHT-40 information is not utilized in training HallNN_VHTver.  

Thrust (HallNN_VHTver)          |  Discharge current (HallNN_VHTver)  
:-------------------------:|:-------------------------:  
![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/results/HallNN_VHTver_KHT40_AFR_Thrust_V250.png?raw=true)  |  ![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/results/HallNN_VHTver_KHT40_AFR_Id_V250.png?raw=true)  


## HallNN training with a virtual Hall thruster dataset
### Training_HallNN_with_VHT_Dataset.m  
Requirements: Deep Learning Toolbox / Curve Fitting Toolbox  

## Simple execution of HallNN
It's as easy as:  
```matlab
[Thrust, sig_Thrust, Id, sig_Id] = HallNN(AFR, Va-Vc, Rout, Rin, Lch, Br_fit_coeff, HallNN_best, flag_disp);
```
Thrust, discharge current, and prediction uncertainties are estimated based on the characteristics of the KHT-40 (FM) 200 W-class Hall thruster.  

### Run_HallNN.m  
Requirements: Deep Learning Toolbox / Curve Fitting Toolbox  

### Run_HallNN_with_fcns.m  
*All network objects were converted into simple functions, located in the +HallNN_fcn folder.*  
Requirements: Curve Fitting Toolbox  

In both cases, the output will be:  
**Thrust = 6.48 mN | sig_Thrust = 0.13 mN | Id = 0.43 A | sig_Id = 0.01 A**  

