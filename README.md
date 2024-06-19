# HallNN: Code and data for "Predicting performance of Hall effect ion source using machine learning"
Hall thruster performance prediction with Neural Network Ensemble

Dataset for HallNN training
=============

![image](https://github.com/JaehongPark-Plasma/HallNN/assets/78528144/dd57ef35-97ee-44f8-8eb3-63fc11026332)

# Information of RandomHallThruster_Dataset.csv
Contains 2,500 randomly generated Hall thruster performance data points for neural network training   

*Rows from 1 to 13* are **INPUT** parameters   
> 1. Anode flow rate (SCCM)
> 2. Voltage drop (V)
> 3. Outer channel radius (mm)
> 4. Inner channel radius (mm)
> 5. Discharge channel length (mm)
> 6 - 13. Radial magnetic field fitting coefficients   

*Rows from 14 and 15* are **THRUST** and **DISCHARGE CURRENT**   
> 14. Thrust (mN)
> 15. Discharge current (A)


*Rows from 16 and 17* are **Standard Deviation** of **THRUST** and **DISCHARGE CURRENT**   
> 14. STD of thrust (mN)
> 15. STD of discharge current (A)


Note) Standard deviation of 'empirical parameter average method' result and which are *not used* in the training process (see the paper).    

# Information of NN_RHT_in.mat and NN_RHT_out.mat
*NN_RHT_in.mat* is the input mat file (13 parameters) for the MATLAB code.   
*NN_RHT_out.mat* is the output dataset (**thrust** and **discharge current**) mat file (13 parameters) for the MATLAB code.   
