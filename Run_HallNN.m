%% J. Park et al., Predicting Performance of Hall Effect Ion Source Using Machine Learning
clc;clear;
disp('Requirement: Deep Learning Toolbox / Curve Fitting Toolbox');
disp(' ');
% Requirements: Deep Learning Toolbox / Curve Fitting Toolbox
% This will load pretrained HallNN's neural networks
load("Pretrained_HallNN.mat");

% Input parameters (#1-5)
% For better interpretation, we seperatly write Va and Vc istead of Vd (voltage drop)
AFR    = 6;     % SCCM, anode (mass) flow rate, we use 0.09763 mg/s/sccm for consistency
Va     = 250;   % V,  anode voltage
Vc     = 30;    % V,  cathode coupling voltage, due to keeper voltage and resistive path from cathode tip to thruster coupling plume
Rout   = 20 ;   % mm, discharge channel outer radius 
Rin    = 13.5 ; % mm, discharge channel inner radius 
Lch    = 25;    % mm, discharge channel length

% Radial Magnetic Field (Br) input coefficient generation
% Br data should be [N x 2] array
% First column: axial position #[mm]# z = 0 @anode <=> at least z = 2*Lch
% Second column: radial magnetic field strength #[T]#, it can be negative or positive
% load radial magnetic field txt data
BrDataName = 'Data/MagneticField_Br/BrData_KHT40.txt';
BrData_raw = readmatrix(BrDataName); 
% Requirements: Curve Fitting Toolbox
fitting_plot = 1; % if 1 == plot Br fitting result, if not satisfied use cotum function fitting method
Br_fit_coeff = B_fit(BrData_raw,Lch,fitting_plot);

% want to display prediction result in the command window?
flag_disp = 1;

% Now it is HallNN function
[Thrust,sig_Thrust,Id,sig_Id] = HallNN(AFR,Va-Vc,Rout,Rin,Lch,Br_fit_coeff,HallNN_best,flag_disp);

%%
function [Thrust,sig_Thrust,Id,sig_Id] = HallNN(AFR,Vd,Rout,Rin,Lch,Br_fit_coeff,HallNN_best,flag_disp)
% If, Isp and Anode efficiency calculation is required, use this
% function [Thrust,sig_Thrust,Id,sig_Id] = HallNN(AFR,Va,Vc,Rout,Rin,Lch,Br_fit_coeff,HallNN_best,flag_disp)

NNens = HallNN_best{1}; % Neural networks 1x100 cell 
net_ens_no = HallNN_best{3}; % Ensembled number = 100
NN_minmax = HallNN_best{9};

% If Isp and Anode efficiency calculation is required, uncomment these
% Vd = Va - Vc; % V_anode - V_cathode coupling voltage
% s2mgs = 0.09763; % 1 sccm to mg/s (used in both HallNN and Numerical simulation for consistency)

for ens = net_ens_no:-1:1
    NN_input = [AFR, Vd, Rout, Rin, Lch, Br_fit_coeff];
    NN_input = log(NN_input); % Logarithmic transformation
    NN_input = (NN_input-NN_minmax.input_min)./NN_minmax.input_max;
    NN_output(1:2) = NNens{ens}(NN_input');
    NN_output(1:2) = exp(NN_output(1:2).*NN_minmax.output_max+NN_minmax.output_min); % Recover-Logarithmic transformation
    thrust_(ens) =  NN_output(1); % mN
    current_(ens) = NN_output(2); % A

    % If Isp and Anode efficiency calculation is required, uncomment these
    % eff_(ens) = (thrust_(ens)^2)/(2*AFR*s2mgs*Va*current_(ens)); % -, anode efficieny, it requires Va to calculate it
    % Isp_(ens) = (thrust_(ens))/(AFR*s2mgs*9.8067)*1e3; % s, anode specific impulse
end

Thrust = mean(thrust_);
sig_Thrust = std(thrust_);
Id = mean(current_);
sig_Id = std(current_);

if (flag_disp == 1)
    disp(['Thrust = ',num2str(round(Thrust,2)),' mN | sig_Thrust = ',num2str(round(sig_Thrust,2)),' mN | Id = ',num2str(round(Id,2)),' A | sig_Id = ',num2str(round(sig_Id,2)),' A']);
end

end
