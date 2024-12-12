%% J. Park et al., Predicting Performance of Hall Effect Ion Source Using Machine Learning
clc; clear; close all;
disp([' ']);
disp(['HallNN Prediction Results with KHT-40, 200 W-class KAIST Hall thruster (FM)']);
disp(['with pre-trained neural networks from the manuscript']);
disp([' ']);
disp(['You can also chose to use Virtual Hall Thruster dataset trained HallNN - "ressults/HallNN_VHTver.mat"'])
disp([' ']);

%% Validation CASE - 1 (HallNN prediction)
% KHT-40 Hall thruster
clc; clear; close all;
SAVE = 1; % if == 1 => fig save.
font = 18;
LW = 1.5;
err1 = 2.576; % 99% CI
err2 = 1.282; % 80% CI
s2mgs = 0.09763; % sccm to mg/s for Xe

BrDataName = 'Data/MagneticField_Br/BrData_KHT40.txt';

load('Data/Experimental/Experimental_KHT40.mat');
load('Data/Numerical/Numerical_KHT40.mat');

% Load manuscript version HallNN
load('Pretrained_HallNN.mat');
NNens = HallNN_best{1};
NNTRens = HallNN_best{2};
net_ens_no = HallNN_best{3};
net_epoch = HallNN_best{7};
NN_minmax = HallNN_best{9};

% Load VHT-dataset-only-trained version HallNN -> Now it is a test case
% load('results/HallNN_VHTver.mat');
% NNens = NN_result{1};
% NNTRens = NN_result{2};
% net_ens_no = NN_result{3};
% net_epoch = NN_result{7};
% NN_minmax = NN_result{9};


%====== THRUST / MFRens
% from NN
MFR    = linspace(6,12,13);    % SCCM
Va     = 250;   % V
Vc     = 30;    % V
Rout   = 20 ;   % mm
Rin    = 13.5 ; % mm
Lch = 25;       % mm
% mag input coeff
BrData_raw = readmatrix(BrDataName); 
fitting_plot = 1; % if 1 == plot Br fitting result
BCoff_K40 = B_fit(BrData_raw,Lch,fitting_plot);
%
% array setting
thrust_ = zeros(net_ens_no,1);
current_ = zeros(net_ens_no,1);
eff_ = zeros(net_ens_no,1);
Isp_ = zeros(net_ens_no,1);
thrust = zeros(numel(MFR),1);
thrust_std = zeros(numel(MFR),1);
current = zeros(numel(MFR),1);
current_std = zeros(numel(MFR),1);
eff = zeros(numel(MFR),1);
eff_std = zeros(numel(MFR),1);
Isp = zeros(numel(MFR),1);
Isp_std = zeros(numel(MFR),1);

tic
MFR_trig = 11; % sccm, for histogram plot
for i=1:numel(MFR)
    for ens = 1:net_ens_no
        NN_input = [MFR(i), Va-Vc, Rout, Rin, Lch, BCoff_K40];
        NN_input = log(NN_input); % Logarithmic transformation
        NN_input = (NN_input-NN_minmax.input_min)./NN_minmax.input_max;
        NN_output(1:2) = NNens{ens}(NN_input');
        NN_output(1:2) = (NN_output(1:2).*NN_minmax.output_max+NN_minmax.output_min);
        NN_output(1:2) = exp(NN_output(1:2)); % Recover-Logarithmic transformation

        thrust_(ens) =  (NN_output(1));
        current_(ens) = (NN_output(2));
        eff_(ens) = (thrust_(ens)^2)/(2*MFR(i)*s2mgs*Va*current_(ens));
        Isp_(ens) = (thrust_(ens))/(MFR(i)*s2mgs*9.8067)*1e3;
    end
    if (MFR(i) == MFR_trig)
        thrust_hist = thrust_;
    end
    thrust(i) = mean(thrust_);
    thrust_std(i) = std(thrust_);
    current(i) = mean(current_);
    current_std(i) = std(current_);
    eff(i) = mean(eff_);
    eff_std(i) = std(eff_);
    Isp(i) = mean(Isp_);
    Isp_std(i) = std(Isp_);
end
toc
% Scaling Law, Lee et al., J. Propul. Power 35, 1073-1079 (2019).
thrust_SC = 892.7*MFR*s2mgs*1e-6*Va^0.5 *1e3;
current_SC = ones(1,numel(MFR)).*633.0*((Rout+Rin))^2 * 1e-6;
eff_SC = (thrust_SC.^2)./(2*MFR*s2mgs*Va.*current_SC);
Isp_SC = (thrust_SC)./(MFR*s2mgs*9.8067)*1e3;
power = Va.*current;

%% Prediction result plot (1) - Anode flow rate / Thrust
% AFR / T NN
j=0;
figure(101)
S(1)=shadedErrorBar(MFR,thrust,err1*thrust_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
hold on;
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,thrust,err2*thrust_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);

P1 = plot(numerical_(:,1),numerical_(:,2),'ko','LineWidth', LW+0.5);
P2 = errorbar(KHT40pfm((KHT40pfm(:,2)==Va),1),KHT40pfm((KHT40pfm(:,2)==Va),13),KHT40pfm((KHT40pfm(:,2)==Va),14),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,thrust_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 25])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Thrust (mN)','Fontsize',font);

legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','HallNN, \mu', 'HallNN, 80% CI', 'HallNN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);
legend('boxoff')

box on;

annotation('arrow', [0.779,0.779], [0.55,0.47],'Color','k','LineWidth',1.5);
annotation('ellipse',[0.7571,0.5524, 0.0446,0.0619],'Color','k','LineStyle',':','LineWidth',2.0);

axes('Position',[.69 .26 0.18 0.2])
box on
Nbins = 8;
H = histogram(thrust_hist,Nbins,'Normalization','pdf');
set(gca,'XMinorTick','off','YMinorTick','off','Fontsize',font-8,'linewidth',LW-0.5)
hold on
T_m = mean(thrust_hist);
T_std = std(thrust_hist);
T = T_m-T_std*5:0.001:T_m+T_std*5;
fT = exp(-(T-T_m).^2./(2*T_std^2))./(T_std*sqrt(2*pi));
plot(T,fT,'LineWidth',LW+0.5)
xlim([T_m-T_std*6 T_m+T_std*6])
ylim([0 max(fT)*1.3])
xlabel('Thrust (mN)','FontSize',font-8);
ylabel('PDF','FontSize',font-8);
set(gca,'ytick',[]);
if (SAVE == 1)
    saveas(gcf,['results/HallNN_KHT40_AFR_Thrust_V',num2str(Va),'.png']);
end
hold off;

%% Prediction result plot (2) - Anode flow rate / Discharge Current
% AFR / Id NN

figure(102)
S(1)=shadedErrorBar(MFR,current,err1*current_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
hold on;
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,current,err2*current_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);

P1 = plot(numerical_(:,1),numerical_(:,2),'ko','LineWidth',LW+0.5);
P2 = errorbar(KHT40pfm((KHT40pfm(:,2)==Va),1),KHT40pfm((KHT40pfm(:,2)==Va),8),KHT40pfm((KHT40pfm(:,2)==Va),9),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,current_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 1.5])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Discharge current (A)','Fontsize',font);

legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','HallNN, \mu', 'HallNN, 80% CI', 'HallNN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')
box on;
if (SAVE == 1)
    saveas(gcf,['results/HallNN_KHT40_AFR_Id_V',num2str(Va),'.png']);
end
hold off;
