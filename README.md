# HallNN: Code and training dataset for "Predicting performance of Hall effect ion source using machine learning"
**Hall** thruster performance prediction with **N**eural **N**etwork ensemble   
Accepted, Advanced Intelligent Systems (2024)
DOI: 10.1002/aisy.202400555

## HallNN structure
### **Ensemble of 100 neural networks**

![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/Data/intro.png?raw=true)

* **13 inputs (thruster design and operating parameters)**
  * Anode flow rate (SCCM)
  * Voltage drop (V)
  * Outer channel radius (mm)
  * Inner channel radius (mm)
  * Discharge channel length (mm)
  * Eight radial magnetic field fitting coefficients
  * (See *Information of Dataset* for details)
* **4 outputs**
  * Thrust & Prediction uncertainty (standard deviation)
  * Discharge current & Prediction uncertainty (standard deviation)

## Information of VirtualHallThruster_Dataset.csv
Contains 4,500 **"virtual"** Hall thruster performance data points, which were generated with numerical simulation (in-house KEPSi-1D code) where Hall thruster design & operating parameters were generated ramdomly.

*Rows from 1 to 13* are **INPUT** parameters   
> 1. Anode flow rate (SCCM), 1 sccm = 0.09763 mg/s relationship was used.
> 2. Voltage drop (V) = Anode voltage - cathode coupling voltage
> 3. Outer channel radius (mm)
> 4. Inner channel radius (mm)
> 5. Discharge channel length (mm)
> 6. to 13. Radial magnetic field fitting coefficients

*Rows from 14 and 15* are **THRUST** and **DISCHARGE CURRENT**   
> 14. Thrust (mN)
> 15. Discharge current (A)

*Rows from 16 and 17* are **Standard Deviation** of **THRUST** and **DISCHARGE CURRENT** from numerical simulations.
> 14. STD of thrust (mN)
> 15. STD of discharge current (A)

Note) Standard deviations of 'empirical parameter average method' result, which were *not used* in the training process (see the paper).    

## Information of NN_VHT_in.mat and NN_VHT_out.mat
Contains 4,500 **"virtual"** Hall thruster performance data points.   
*NN_VHT_in.mat* is the input mat file (13 parameters) for the MATLAB code.   
*NN_VHT_out.mat* is the output mat file (**thrust** and **discharge current**) for the MATLAB code.   

## Vritual Hall Thruster dataset preview
### Input parameters $\leftrightarrow$ Thrust
![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/Data/Input_thrust_VHT.png?raw=true)
### Input parameters $\leftrightarrow$ Discharge current
![image](https://github.com/JaehongPark-Plasma/HallNN/blob/main/Data/Input_Id_VHT.png?raw=true)

## Prediction using Trained Neural Networks (manuscript version)

## Training using Virtual Hall Thruster dataset
