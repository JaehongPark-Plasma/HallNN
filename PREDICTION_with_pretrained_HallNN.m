%% Plot randomly generated Hall thruster dataset
% Input <=> T
clc;clear;
load('Data/DATA_RHT_in.mat');
load('Data/DATA_RHT_out.mat');
LW = 0.7;
font = 17;
figure(703)
t = tiledlayout(4,5,'TileSpacing','Compact');
% 1
nexttile
plot(rinput(:,1),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('Anode flow rate (sccm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
% 2
nexttile
plot(rinput(:,2),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('V_d (V)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,2))*0.8 max(rinput(:,2))*1.05])
% 3
nexttile
plot(rinput(:,3),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('R_{out} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,3))*0.8 max(rinput(:,3))*1.05])
% 4
nexttile
plot(rinput(:,4),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('R_{in} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,4))*0.8 max(rinput(:,4))*1.05])
% 5
nexttile
plot(rinput(:,5),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('L_{ch} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,5))*0.8 max(rinput(:,5))*1.05])
% 6
nexttile
plot(rinput(:,6),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('B_m (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,6))*0.8 max(rinput(:,6))*1.05])
% 7
nexttile
plot(rinput(:,7),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('L_m (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,7))*0.9 max(rinput(:,7))*1.05])
% 8
nexttile
plot(rinput(:,8),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('B_1 (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,8))*0.9 max(rinput(:,8))*1.05])
% 9
nexttile(11)
plot(rinput(:,9),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('c_1 (-)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,9))*0.9 max(rinput(:,9))*1.05])
% 10
nexttile(12)
plot(rinput(:,10),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('s_1 (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,10))*0.9 max(rinput(:,10))*1.05])
% 11
nexttile(13)
plot(rinput(:,11),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('B_2 (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,11))*0.9 max(rinput(:,11))*1.05])
% 12
nexttile(16)
plot(rinput(:,12),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('c_2 (-)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,12))*0.9 max(rinput(:,12))*1.05])
% 13
nexttile(17)
plot(rinput(:,13),routput(:,1),'.','color','#FF4551','Markersize',4,'linewidth',LW-0.5);
xlabel('s_2 (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,13))*0.9 max(rinput(:,13))*1.05])

leg = legend('Random Hall thruster','Fontsize',font-4,'location','northwest');
leg.Layout.Tile = 18;

ylabel(t,'Thrust (mN)','Fontsize',font+3)

x0=50;
y0=50;
width=1600;
height=900;
set(gcf,'position',[x0,y0,width,height])
hold off;
saveas(gcf,'results/Dataset_Input_Thrust.png');

%histogram
RanHall_P = rinput(:,2).*routput(:,2);
RanHall_T = routput(:,1);

PP = [RanHall_P];
TT = [RanHall_T];
RK_name = cell(numel(PP),1);
for i=1:numel(RanHall_P)
RK_name{i} = 'RHT';
end
figure(20)
c = scatterhist(PP,TT,'Group',RK_name,'Kernel','on','Style','bar','NBins',[30,30],'Location','SouthWest','Direction','in','Color','rb','LineStyle',{'-','-'},'LineWidth',1.5,'Marker','.','MarkerSize',4.5,'legend','off');

hold on;
dd = rinput(:,2).*routput(:,2);
LL = linspace(0,1e5,100);
RH_TP = mean(routput(:,1)./(rinput(:,2).*routput(:,2)))*1e3;
disp(['Dataset T/P  :  ',num2str(RH_TP),'  mN/kW'])
plot(LL,LL*40e-3,'--k','linewidth',LW+2.5);
plot(LL,LL*80e-3,'--k','linewidth',LW+2.5);
xlabel('V_d\cdotI_d (W)','Fontsize',font)
ylabel('Thrust (mN)','Fontsize',font)
text(900,150,'80 mN/kW','Fontsize',font-4)
text(2000,65,'40 mN/kW','Fontsize',font-4)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW+0,'Layer','top')
xlim([0 3000])
ylim([0 200])
hold off;
x0=50;
y0=50;
width=600;
height=600;
set(gcf,'position',[x0,y0,width,height])
saveas(gcf,'results/Dataset_Input_Thrust_Histogram.png');

% =========================================================
disp([' ']);
disp(['HallNN Prediction Results with KHT-40']);
disp(['with pre-trained NN from RHT 2,500 points']);

%% Test CASE - 1 (HallNN prediction)
% Originally, it is a validation case but, in this HallNN open-version it is the test case.
% KHT-40 Hall thruster
clc; clear; close all;
SAVE = 1; % if == 1 => fig save.
font = 18;
LW = 1.5;
err1 = 2.576; % 99% CI
err2 = 1.282; % 80% CI

BrDataName = 'Data/MagneticField_Br/BrData_KHT40.txt'

load('Data/Experimental/Experimental_KHT40.mat');
load('Data/Numerical/Numerical_KHT40.mat');
load('PRETRAINED_HallNN_open.mat');

NNens = NN_result{1};
NNTRens = NN_result{2};
net_ens_no = NN_result{3};
net_epoch = NN_result{7};
s2mgs = 0.09763;

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
        NN_input = (NN_input-NN_minmax.input_min)./NN_minmax.input_max;
        NN_output(1:2) = NNens{ens}(NN_input');
        NN_output(1:2) = (NN_output(1:2).*NN_minmax.output_max+NN_minmax.output_min);
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
