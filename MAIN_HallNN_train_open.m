%% reload
clc;clear;
% load training dataset
load("Data/DATA_RHT_in.mat"); % input
load("Data/DATA_RHT_out.mat"); % output

% Normalize the training dataset
input_c = (rinput);
output_c = (routput);

input_min = min(input_c);
input_ = input_c-input_min;
input_max = max(input_);
input_norm = input_./input_max;

output_min = min(output_c);
output_ = output_c-output_min;
output_max = max(output_);
output_norm = output_./output_max;

input_norm = input_norm';
output_norm = output_norm';
% Now, data is --> [0 1] normalized

% NN_minmax: store the min/max of input/output
NN_minmax.input_min = input_min;
NN_minmax.input_max = input_max;
NN_minmax.output_min = output_min;
NN_minmax.output_max = output_max;


%% Neural Network Training
% Neural netowrk's hyperparameters
% epsilon: Adversarial training rate
eps_adv = 0.1; % percent
% M: Number of ensembled networks (NN_study cell storage)
net_ens_no = 150;
% H1: Hidden layer 1 node numer
HL1size = 20;
% H2: Hidden layer 2 node numer
HL2size = 8;

% Early stop setting
net_epoch = 50;
% Test case fraction
Test_frac = 0.1;

% preallocation
% Neural Network Ensemble cell
NNens = cell(1,net_ens_no);
% Neural Network Train Result Ensemble cell
NNTRens = cell(1,net_ens_no);

% Parallel trainings for M networks
parfor ens_i = 1:net_ens_no
    % Adversarial training dataset -> regenerate at each single NN training
    input_norm_adv = input_norm.*( 1 + eps_adv*1e-2*(2*rand(numel(input_norm(:,1)),numel(input_norm(1,:)))-1));
    % Final training dataset for each NNs
    input_NN_ = [input_norm, input_norm_adv];
    % output of adversarial dataset is same as original ouput dataset
    output_NN_ = [output_norm, output_norm];

    % DATA SHUFFLE 
    % -> for 90% random sampling of training dataset / 10% test dataset
    size = numel(input_NN_(1,:));
    ix = randperm(size);
    input_NN = input_NN_(:,ix);
    output_NN = output_NN_(:,ix);
    data_num = size;

    % Single NN setting
    net = fitnet([HL1size HL2size]);
    net.performFcn = 'mse';
    net.trainFcn = 'trainbr'; % Bayesian regularization backpropagation 
    % Remove bias and normalize the dataset
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
    
    % NN's activation functions
    net.layers{1}.transferFcn = 'tansig'; % Tangent sigmoid
    net.layers{2}.transferFcn = 'tansig'; % Tangent sigmoid
    net.layers{3}.transferFcn = 'purelin';% linear

    % Early stop setting
    net.trainParam.epochs = net_epoch;

    % Turn off the GUI training screen
    net.trainParam.showWindow = false;  
    
    % Divide targets into three sets using interleaved indices
    net.divideFcn = 'divideind'; 
    test_num = round(data_num*Test_frac);
    % Training dataset indices
    net.divideParam.trainInd = 1:(data_num-(test_num));
    % Test dataset indices
    net.divideParam.testInd  = (data_num-(test_num))+1:(data_num);

    % Train!
    [NNens{ens_i}, NNTRens{ens_i}] = train(net,input_NN,output_NN,'useParallel','no'); % Do not use parallel training for indiviual NN trains.
    % Current networks / total M | Train MSE | Test MSE
    disp(['Training :: ', num2str(ens_i),' / ',num2str(net_ens_no),' | train: ' ,num2str(NNTRens{ens_i}.perf(end)),' | test:' ,num2str(NNTRens{ens_i}.tperf(end))]);
end
% save the case study result
NN_result{1} = NNens; % Trained networks
NN_result{2} = NNTRens; % Trained result (MSE,...)
NN_result{3} = net_ens_no; % M
NN_result{4} = HL1size; % H1
NN_result{5} = HL2size; % H2
NN_result{6} = eps_adv; % epsilon
NN_result{7} = net_epoch; % Early stop
NN_result{8} = Test_frac; % Test dataset fraction

%% SAVE case study result
save(['results/HallNN_open.mat'],"NN_result","NN_minmax",'-v7.3');

%% Result, MSE trend
clear;
% select network model
load('results/HallNN_open.mat');
NNTRens = NN_result{2};
net_ens_no = NN_result{3};
net_epoch = NN_result{7};
font = 18;
LW = 1.5;
for i=1:net_ens_no
    TrainPerf(i,1:numel(NNTRens{i}.perf))=NNTRens{i}.perf;
end
TrainPerf = TrainPerf';
for i=1:net_ens_no
    TestPerf(i,1:numel(NNTRens{i}.tperf))=NNTRens{i}.tperf;
end
TestPerf = TestPerf';

for j=1:net_epoch+1
    kk = 0;
    jj = 0;
    clear tempTrain tempTest;
    for i=1:net_ens_no
        if(TrainPerf(j,i) > 0)
            kk = kk+1;
            tempTrain(kk) = TrainPerf(j,i);
        end
        if(TestPerf(j,i) > 0)
            jj = jj+1;
            tempTest(jj) = TestPerf(j,i);
        end
    end
    Train_max(j) = max(tempTrain);
    Train_min(j) = min(tempTrain);
    Train_mean(j) = mean(tempTrain);
    Test_max(j) = max(tempTest);
    Test_min(j) = min(tempTest);
    Test_mean(j) = mean(tempTest);
end

disp(['Result :: train: ' ,num2str(Train_mean(end)),' | test:' ,num2str(Test_mean(end))]);

figure(50)
it = linspace(0,net_epoch,net_epoch+1);
semilogy(it, Train_min,'-w','linewidth',0.5);
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
hold on;
semilogy(it, Train_max,'-w','linewidth',0.5);
x2 = [it, fliplr(it)];
inBetween = [Train_min, fliplr(Train_max)];
h1 = fill(x2, inBetween, 'b','LineStyle','none');
set(h1,'facealpha',.2)
P1 = semilogy(it, Train_mean,' --b','linewidth',LW+0.5);

semilogy(it, Test_min,'-w','linewidth',0.5);
semilogy(it, Test_max,'-w','linewidth',0.5);
P2 = semilogy(it, Test_mean,'-r','linewidth',LW+0.5);
x3 = [it, fliplr(it)];
inBetween3 = [Test_min, fliplr(Test_max)];
h2 = fill(x3, inBetween3, 'r','LineStyle','none');
set(h2,'facealpha',.2)
ylim([1e-5 1e1])
ylabel('Mean squared error (MSE)','Fontsize',font);
xlabel('Epochs (iterations)','Fontsize',font);
legend([P1, P2, h1, h2], 'Mean MSE of training dataset', 'Mean MSE of test dataset', 'MSE envelopes of training dataset', 'MSE envelopes of test dataset','Fontsize',font-5); 
legend('boxoff')

hold off;
saveas(gcf,['results/NN_perf','.png']);

%% TEST
% Originally, it is a validation case but, in this open-version it is the
% test case.
% Figure 10 new
% =================
% KHT-40

clc; clear; close all;
SAVE = 1; % if == 1 => fig save.
font = 18;
LW = 1.5;
err1 = 2.576; % 99% CI
err2 = 1.282; % 80% CI

BrDataName = 'GDPL_Br\BrData_KHT40.txt'

load('Exp_data/KHT40_pfm.mat');
res1 = csvread('Data/ver2021data/RESULT_KHT-40.csv',0);
res2 = csvread('Data/ver2021data/RESULT_KHT-40_2.csv',0);
res2(:,59:60) = 0;
res = [res1(:,1:70); res2;];

load('NN_open_26_Jul_2023_112317');

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
Rout     = 20 ; % mm
Rin     = 13.5 ; % mm
Lch = 25;    % mm
MS = 0;
MX = 1;
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
MFR_trig = 11; % sccm
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

thrust_SC = 892.7*MFR*s2mgs*1e-6*Va^0.5 *1e3;
current_SC = ones(1,numel(MFR)).*633.0*((Rout+Rin))^2 * 1e-6;
eff_SC = (thrust_SC.^2)./(2*MFR*s2mgs*Va.*current_SC);
Isp_SC = (thrust_SC)./(MFR*s2mgs*9.8067)*1e3;
power = Va.*current;

% =================
% Fig.10.a (old)
% MFR / T NN
j=0;
figure(61)
for i=1:numel(res(:,1))
    if(res(i,2)==Va && res(i,3)==Lch && res(i,4)==MS && res(i,5)==MX )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,6);
        temp(j,3) = res(i,11);
    end
end
S(1)=shadedErrorBar(MFR,thrust,err1*thrust_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
hold on; 
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,thrust,err2*thrust_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth', LW+0.5);
end
P2 = errorbar(KHT40pfm((KHT40pfm(:,2)==Va),1),KHT40pfm((KHT40pfm(:,2)==Va),13),KHT40pfm((KHT40pfm(:,2)==Va),14),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,thrust_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 25])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Thrust (mN)','Fontsize',font);
ylt = yl.Position;
yl.Position = [5.3 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
if (SAVE == 1)
saveas(gcf,['Figure_open/Fig_10a_OLD_KHT40_MT_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_10a_OLD_KHT40_MT_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_10a_OLD_KHT40_MT_V',num2str(Va),'.svg']);
end
hold off;

% =================
% Fig.10.b
% MFR / Id NN
clear temp temp2 yl;
j=0;
figure(62)
for i=1:numel(res(:,1))
    if(res(i,2)==Va && res(i,3)==Lch && res(i,4)==MS && res(i,5)==MX )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,8);
        temp(j,3) = res(i,13);
    end
end

S(1)=shadedErrorBar(MFR,current,err1*current_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
  hold on;
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,current,err2*current_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth',LW+0.5);
end
P2 = errorbar(KHT40pfm((KHT40pfm(:,2)==Va),1),KHT40pfm((KHT40pfm(:,2)==Va),8),KHT40pfm((KHT40pfm(:,2)==Va),9),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,current_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 1.5])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Discharge current (A)','Fontsize',font);
yl.HorizontalAlignment='center';
ylt = yl.Position;
yl.Position = [5.3 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
if (SAVE == 1)
saveas(gcf,['Figure_open/Fig_10b_KHT40_MID_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_10b_KHT40_MId_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_10b_KHT40_MId_V',num2str(Va),'.svg']);
end
hold off;

% =================
% Fig.10.c
% MFR / Isp NN
clear temp temp2;
j=0;
figure(63)
for i=1:numel(res(:,1))
    if(res(i,2)==Va && res(i,3)==Lch && res(i,4)==MS && res(i,5)==MX )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,7);
        temp(j,3) = res(i,12);
    end
end
S(1)=shadedErrorBar(MFR,Isp,err1*Isp_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
  hold on;
S(2)=shadedErrorBar(MFR,Isp,err2*Isp_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth',LW+0.5);
end
P2 = errorbar(KHT40pfm((KHT40pfm(:,2)==Va),1),KHT40pfm((KHT40pfm(:,2)==Va),21),KHT40pfm((KHT40pfm(:,2)==Va),22),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,Isp_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 3000])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Anode specific impulse (s)','Fontsize',font);
ylt = yl.Position;
yl.Position = [5.3 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
if (SAVE == 1)
saveas(gcf,['Figure_open/Fig_10c_KHT40_MIsp_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_10c_KHT40_MIsp_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_10c_KHT40_MIsp_V',num2str(Va),'.svg']);
end
hold off;

% =================
% Fig.6.d
% MFR / Eff NN
clear temp temp2;
j=0;
figure(64)
for i=1:numel(res(:,1))
    if(res(i,2)==Va && res(i,3)==Lch && res(i,4)==MS && res(i,5)==MX )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,10);
        temp(j,3) = res(i,15);
    end
end
S(1)=shadedErrorBar(MFR,eff*100,err1*eff_std*100,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
  hold on;
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,eff*100,err2*eff_std*100,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth',LW+0.5);
end
P2 = errorbar(KHT40pfm((KHT40pfm(:,2)==Va),1),KHT40pfm((KHT40pfm(:,2)==Va),18)*100,KHT40pfm((KHT40pfm(:,2)==Va),19)*100,'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,eff_SC*100,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 100])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Anode efficiency (%)','Fontsize',font);
ylt = yl.Position;
yl.Position = [5.3 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
if (SAVE == 1)
saveas(gcf,['Figure_open/Fig_10d_KHT40_MEff_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_10d_KHT40_MEff_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_10d_KHT40_MEff_V',num2str(Va),'.svg']);
end
hold off;

% histogram of KHT-40 @ 250 V, 11 sccm
figure(66)
H = histogram(thrust_hist,'Normalization','pdf');
Nbins = morebins(H);
set(gca,'XMinorTick','off','YMinorTick','off','Fontsize',font-8,'linewidth',LW-0.5)
hold on
T_m = mean(thrust_hist);
T_std = std(thrust_hist);
T = T_m-T_std*5:0.001:T_m+T_std*5;
fT = exp(-(T-T_m).^2./(2*T_std^2))./(T_std*sqrt(2*pi));
plot(T,fT,'LineWidth',LW+0.5)
xlim([T_m-T_std*6 T_m+T_std*6])
ylim([0 max(fT)*1.2])
xlabel('Thrust (mN)','FontSize',font-8);
ylabel('PDF','FontSize',font-8);
x0=100;
y0=100;
width=140;
height=130;
set(gcf,'position',[x0,y0,width,height])
set(gca,'ytick',[]);
if (SAVE == 1)
saveas(gcf,['Figure_open/Fig_10x_Histo.fig']);
saveas(gcf,['Figure_open/Fig_10x_Histo.png']);
saveas(gcf,['Figure_open/Fig_10x_Histo.svg']);
end
hold off;

% =================
% NEW Fig.6.a
% MFR / T NN
j=0;
figure(67)
for i=1:numel(res(:,1))
    if(res(i,2)==Va && res(i,3)==Lch && res(i,4)==MS && res(i,5)==MX )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,6);
        temp(j,3) = res(i,11);
    end
end
S(1)=shadedErrorBar(MFR,thrust,err1*thrust_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
hold on; 
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,thrust,err2*thrust_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth', LW+0.5);
end
P2 = errorbar(KHT40pfm((KHT40pfm(:,2)==Va),1),KHT40pfm((KHT40pfm(:,2)==Va),13),KHT40pfm((KHT40pfm(:,2)==Va),14),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,thrust_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 25])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Thrust (mN)','Fontsize',font);
ylt = yl.Position;
yl.Position = [5.3 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];

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
saveas(gcf,['Figure_open/Fig_10a_KHT40_MT_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_10a_KHT40_MT_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_10a_KHT40_MT_V',num2str(Va),'.svg']);
end
hold off;


%% TEST
%% Figure 11 with SCLAW
% =================
% Fig.11
% KHT-70 Br 2.4
clc; clear; close all;
font = 18;
LW = 1.5;
err1 = 2.576; % 99% CI
err2 = 1.282; % 80% CI

BrDataName = 'GDPL_Br\BrData_KHT70_2.4A.txt'

load('Exp_data/KHT70_24_pfm.mat');
res1 = csvread('Data/ver2021data/RESULT_KHT-70_2_4A.csv',0);
res = res1;

load('NN_open_8_Aug_2023_193939');
NNens = NN_result{1};
NNTRens = NN_result{2};
net_ens_no = NN_result{3};
net_epoch = NN_result{7};
s2mgs = 0.09763;

K70_MTI_20 = [18 1.39	26.67765	0.32962
20	30.89297 1.55	0.3184
22	33.99016 1.72	0.32568
24	36.03554 2.12	0.56728];

K70_MTI_24 = [18 25.19225 1.64 0.18995
20	30.40335 1.63 0.79375
22	32.44992 1.79 0.50623
24	38.52579 2.06 0.58518
26	42.27398 2.38 0.44165];

K70_MTI_28 = [18	23.8632 1.45 0.43683
20	27.51164 1.57 0.34545
22	30.43121 1.71 0.261];

K70_ME_20 = [18	48.64117 1.19821
20	52.97904 0.66597
22	51.93219 0.9939
24	43.47952 1.94842];

K70_ME_24 = [18	37.23211 0.69545
20	49.0274 2.77018
22	44.86777 0.2829
24	50.76427 0.6599
26	49.02551 0.99821];

K70_ME_28 = [18	36.87232 0.64796
20	40.82404 0.46894
22	41.9834 0.51094];

K70_MIsp_20 = [18	1550.41327	19.19289
20	1621.13392	10.22224
22	1612.08335	15.4167
24	1567.65081	35.06719];

K70_MIsp_24 = [18	1473.55397	13.73849
20	1598.62939	45.23801
22	1528.67114	4.8173
24	1670.03779	10.87391
26	1694.85283	17.22321];

K70_MIsp_28 = [18	1378.8034	12.15478
20	1432.24294	8.23361
22	1441.05693	8.7558];

%====== THRUST / MFR
% from NN
MFR    = linspace(16,28,13);    % SCCM
Va     = 300;   % V
Vc     = 20;    % V
Rout     = 36; % mm
Rin     = 22; % mm
Lch = 28;    % mm
% mag input coeff
BrData_raw = readmatrix(BrDataName); 
fitting_plot = 1; % if 1 == plot Br fitting result
BCoff_K70 = B_fit(BrData_raw,Lch,fitting_plot)
%BCoff_K70 = BCoff_K70.*(0.99 + (0.02).*rand(1,8))
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
MFR_trig = 26; % sccm
for i=1:numel(MFR)
    for ens = 1:net_ens_no
        NN_input = [MFR(i), Va-Vc, Rout, Rin, Lch, BCoff_K70];
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

thrust_SC = 892.7*MFR*s2mgs*1e-6*Va^0.5 *1e3;
current_SC = ones(1,numel(MFR)).*633.0*((Rout+Rin))^2 * 1e-6;
eff_SC = (thrust_SC.^2)./(2*MFR*s2mgs*Va.*current_SC);
Isp_SC = (thrust_SC)./(MFR*s2mgs*9.8067)*1e3;
power = Va.*current;

% =================
% OLD Fig.11.a_SC
% MFR / T NN
j=0;
figure(71)
for i=1:numel(res(:,1))
    if(res(i,2)==Va )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,3);
        temp(j,3) = res(i,8);
    end
end
S(1)=shadedErrorBar(MFR,thrust,err1*thrust_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
hold on; 
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,thrust,err2*thrust_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth', LW+0.5);
end
P2 = errorbar(K70_MTI_24(:,1),K70_MTI_24(:,2),K70_MTI_24(:,4),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,thrust_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 60])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Thrust (mN)','Fontsize',font);
ylt = yl.Position;
yl.Position = [14.6 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
%legend([P2, P1, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','\mu, Neural network', '\sigma, Neural network', '2\sigma, Neural network','Location', 'Northwest','Fontsize',font-5);
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
saveas(gcf,['Figure_open/Fig_11a_SC_OLD_KHT70_MT_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_11a_SC_OLD_KHT70_MT_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_11a_SC_OLD_KHT70_MT_V',num2str(Va),'.svg']);
hold off;

% =================
% Fig.11.b_SC
% MFR / Id NN
clear temp temp2;
j=0;
figure(72)
for i=1:numel(res(:,1))
    if(res(i,2)==Va )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,5);
        temp(j,3) = res(i,10);
    end
end

S(1)=shadedErrorBar(MFR,current,err1*current_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
  hold on;
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,current,err2*current_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth',LW+0.5);
end
P2 = errorbar(K70_MTI_24(:,1),K70_MTI_24(:,3),[0;0;0;0;0],'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,current_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 4])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Discharge current (A)','Fontsize',font);
ylt = yl.Position;
yl.Position = [14.6 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
%legend([P2, P1, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','\mu, Neural network', '\sigma, Neural network', '2\sigma, Neural network','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
saveas(gcf,['Figure_open/Fig_11b_SC_KHT70_MID_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_11b_SC_KHT70_MId_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_11b_SC_KHT70_MId_V',num2str(Va),'.svg']);
hold off;

% =================
% Fig.11.c_SC
% MFR / Isp NN
clear temp temp2;
j=0;
figure(73)
for i=1:numel(res(:,1))
    if(res(i,2)==Va )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,4);
        temp(j,3) = res(i,9);
    end
end
S(1)=shadedErrorBar(MFR,Isp,err1*Isp_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
  hold on;
S(2)=shadedErrorBar(MFR,Isp,err2*Isp_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth',LW+0.5);
end
P2 = errorbar(K70_MIsp_24(:,1),K70_MIsp_24(:,2),K70_MIsp_24(:,3),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,Isp_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 3000])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Anode specific impulse (s)','Fontsize',font);
ylt = yl.Position;
yl.Position = [14.6 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
%legend([P2, P1, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','\mu, Neural network', '\sigma, Neural network', '2\sigma, Neural network','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
saveas(gcf,['Figure_open/Fig_11c_SC_KHT70_MIsp_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_11c_SC_KHT70_MIsp_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_11c_SC_KHT70_MIsp_V',num2str(Va),'.svg']);
hold off;

% =================
% Fig.11.d_SC
% MFR / Eff NN
clear temp temp2;
j=0;
figure(74)
for i=1:numel(res(:,1))
    if(res(i,2)==Va )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,7);
        temp(j,3) = res(i,12);
    end
end
S(1)=shadedErrorBar(MFR,eff*100,err1*eff_std*100,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
  hold on;
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,eff*100,err2*eff_std*100,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth',LW+0.5);
end
P2 = errorbar(K70_ME_24(:,1),K70_ME_24(:,2),K70_ME_24(:,3),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,eff_SC*100,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 100])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Anode efficiency (%)','Fontsize',font);
ylt = yl.Position;
yl.Position = [14.6 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
%legend([P2, P1, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','\mu, Neural network', '\sigma, Neural network', '2\sigma, Neural network','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
saveas(gcf,['Figure_open/Fig_11d_SC_KHT70_MEff_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_11d_SC_KHT70_MEff_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_11d_SC_KHT70_MEff_V',num2str(Va),'.svg']);
hold off;

% =================
% NEW Fig.11.a_SC
% MFR / T NN
figure(77)
j=0;
for i=1:numel(res(:,1))
    if(res(i,2)==Va )
        j=j+1;
        temp(j,1) = res(i,1);
        temp(j,2) = res(i,3);
        temp(j,3) = res(i,8);
    end
end
S(1)=shadedErrorBar(MFR,thrust,err1*thrust_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
hold on; 
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,thrust,err2*thrust_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
if (j ~= 0)
P1 = plot(temp(:,1),temp(:,2),'ko','LineWidth', LW+0.5);
end
P2 = errorbar(K70_MTI_24(:,1),K70_MTI_24(:,2),K70_MTI_24(:,4),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,thrust_SC,'k+', 'LineWidth', LW+0.5);
xlim([min(MFR) max(MFR)])
ylim([0 60])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Thrust (mN)','Fontsize',font);
ylt = yl.Position;
yl.Position = [14.6 ylt(2:3)];
if (j ~= 0)
legend([P2, P1, P3, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','Scaling law','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
%legend([P2, P1, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Numerical','\mu, Neural network', '\sigma, Neural network', '2\sigma, Neural network','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
else
legend([S(1).mainLine, S.patch, P2],'\mu,   Neural network', '2\sigma, Neural network', '\sigma,   Neural network','Experimental','Location', 'Northwest','Fontsize',font-5); 
legend('boxoff')
end
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];

annotation('arrow', [0.779,0.779], [0.6227,0.4881],'Color','k','LineWidth',1.5);
annotation('ellipse',[0.7571,0.619, 0.0446,0.0619],'Color','k','LineStyle',':','LineWidth',2);

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
xticks([36,38,40,42])
saveas(gcf,['Figure_open/Fig_11a_SC_KHT70_MT_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_11a_SC_KHT70_MT_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_11a_SC_KHT70_MT_V',num2str(Va),'.svg']);
hold off;

%% Figure 12(?) with SCLAW
% =================
% Fig.12
% KHT-100v2 Cox Measurement 200 G
clear; close all;
font = 18;
LW = 1.5;
err1 = 2.576; % 99% CI
err2 = 1.282; % 80% CI

BrDataName = 'GDPL_Br\BrData_KHT100v2_cox_meas.txt'

load('NN_open_26_Jul_2023_112317');
NNens = NN_result{1};
NNTRens = NN_result{2};
net_ens_no = NN_result{3};
net_epoch = NN_result{7};
s2mgs = 0.09763;

% Va (1) Id(2) P(3) Xe(4) T(5) err(6) Isp(7) err(8) eff.(9) err (10) T/P(11) err(12)
K100v2C_200G = [300	3.03	909	34	55.07879	0.35332	1692.00439	10.85391	50.27148	0.64598	60.59273	0.38869
300	3.17	951	36	56.52186	0.11205	1639.87216	3.251	47.79021	0.18942	59.43413	0.11783
300	3.4	1020	38	61.50525	0.66556	1690.53682	18.09051	49.98714	1.08406	60.29927	0.65251
300	3.57	1071	40	64.09096	0.0783	1673.52725	2.27162	49.10564	0.12577	59.84216	0.07311
300	3.71	1113	42	68.38693	0.41852	1700.6692	10.40778	51.23889	0.62772	61.44378	0.37602
300	3.91	1173	44	71.77062	0.95283	1703.6879	22.61816	51.11992	1.359	61.18552	0.8123
300	4.16	1248	46	77.81175	0.03402	1766.78377	0.77243	54.0138	0.04723	62.34916	0.02726];


%====== THRUST / MFR
% from NN
MFR    = linspace(30,50,11);    % SCCM
Va     = 300;   % V
Vc     = 10;    % V
Rout     = 50; % mm
Rin     = 35; % mm
Lch = 25;    % mm
% mag input coeff
BrData_raw = readmatrix(BrDataName); 
BrData_raw(:,1) = BrData_raw(:,1) - 0.1;
fitting_plot = 1; % if 1 == plot Br fitting result
BCoff_K100C = B_fit(BrData_raw,Lch,fitting_plot);
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
MFR_trig = 46; % sccm
for i=1:numel(MFR)
    for ens = 1:net_ens_no
        NN_input = [MFR(i), Va-Vc, Rout, Rin, Lch, BCoff_K100C];
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
        Id_hist = current_;
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

thrust_SC = 892.7*MFR*s2mgs*1e-6*Va^0.5 *1e3;
current_SC = ones(1,numel(MFR)).*633.0*((Rout+Rin))^2 * 1e-6;
eff_SC = (thrust_SC.^2)./(2*MFR*s2mgs*Va.*current_SC);
Isp_SC = (thrust_SC)./(MFR*s2mgs*9.8067)*1e3;
power = Va.*current;

% =================
ylab_pos = 27.6;
% OLD Fig.12.a_SC
% MFR / T NN
j=0;
figure(121)
S(1)=shadedErrorBar(MFR,thrust,err1*thrust_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
hold on; 
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,thrust,err2*thrust_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
P2 = errorbar(K100v2C_200G(:,4),K100v2C_200G(:,5),K100v2C_200G(:,6),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,thrust_SC,'k+', 'LineWidth', LW+0.5);
Pn = plot(NaN,NaN,'o', 'Marker', 'none');
xlim([min(MFR) max(MFR)])
ylim([30 100])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Thrust (mN)','Fontsize',font);
ylt = yl.Position;
yl.Position = [ylab_pos ylt(2:3)];
legend([P2, P3, Pn, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Scaling law','','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')

box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
saveas(gcf,['Figure_open/Fig_12a_SC_OLD_KHT100v2c_MT_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_12a_SC_OLD_KHT100v2c_MT_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_12a_SC_OLD_KHT100v2c_MT_V',num2str(Va),'.svg']);
hold off;
%
% =================
% Fig.12.b_SC
% MFR / Id NN
clear temp temp2;
j=0;
figure(122)
S(1)=shadedErrorBar(MFR,current,err1*current_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
  hold on;
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,current,err2*current_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
P2 = errorbar(K100v2C_200G(:,4),K100v2C_200G(:,2),[0;0;0;0;0;0;0],'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,current_SC,'k+', 'LineWidth', LW+0.5);
Pn = plot(NaN,NaN,'o', 'Marker', 'none');
xlim([min(MFR) max(MFR)])
ylim([2 6])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Discharge current (A)','Fontsize',font);
ylt = yl.Position;
yl.Position = [ylab_pos ylt(2:3)];
legend([P2, P3, Pn, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Scaling law','','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')

box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
saveas(gcf,['Figure_open/Fig_12b_SC_KHT100v2c_MID_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_12b_SC_KHT100v2c_MId_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_12b_SC_KHT100v2c_MId_V',num2str(Va),'.svg']);
hold off;

% =================
% Fig.12.c_SC
% MFR / Isp NN
clear temp temp2;
j=0;
figure(73)
S(1)=shadedErrorBar(MFR,Isp,err1*Isp_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
  hold on;
S(2)=shadedErrorBar(MFR,Isp,err2*Isp_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
P2 = errorbar(K100v2C_200G(:,4),K100v2C_200G(:,7),K100v2C_200G(:,8),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,Isp_SC,'k+', 'LineWidth', LW+0.5);
Pn = plot(NaN,NaN,'o', 'Marker', 'none');
xlim([min(MFR) max(MFR)])
ylim([0 3000])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Anode specific impulse (s)','Fontsize',font);
ylt = yl.Position;
yl.Position = [ylab_pos ylt(2:3)];
legend([P2, P3, Pn, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Scaling law','','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')

box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
saveas(gcf,['Figure_open/Fig_12c_SC_KHT100v2c_MIsp_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_12c_SC_KHT100v2c_MIsp_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_12c_SC_KHT100v2c_MIsp_V',num2str(Va),'.svg']);
hold off;

% =================
% Fig.12.d_SC
% MFR / Eff NN
clear temp temp2;
j=0;
figure(124)
S(1)=shadedErrorBar(MFR,eff*100,err1*eff_std*100,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
  hold on;
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,eff*100,err2*eff_std*100,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
P2 = errorbar(K100v2C_200G(:,4),K100v2C_200G(:,9),K100v2C_200G(:,10),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,eff_SC*100,'k+', 'LineWidth', LW+0.5);
Pn = plot(NaN,NaN,'o', 'Marker', 'none');
xlim([min(MFR) max(MFR)])
ylim([0 100])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Anode efficiency (%)','Fontsize',font);
ylt = yl.Position;
yl.Position = [ylab_pos ylt(2:3)];
legend([P2, P3, Pn, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Scaling law','','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')

box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];
saveas(gcf,['Figure_open/Fig_12d_SC_KHT100v2c_MEff_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_12d_SC_KHT100v2c_MEff_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_12d_SC_KHT100v2c_MEff_V',num2str(Va),'.svg']);
hold off;

% =================
% NEW Fig.12.a_SC
% MFR / T NN
figure(127)
j=0;
S(1)=shadedErrorBar(MFR,thrust,err1*thrust_std,'lineProps',{'color', [255 141 141]/255,'linewidth',LW-0.5},'transparent',0);
hold on; 
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
S(2)=shadedErrorBar(MFR,thrust,err2*thrust_std,'lineProps',{'color','r','linewidth',LW-0.5},'transparent',0);
P2 = errorbar(K100v2C_200G(:,4),K100v2C_200G(:,5),K100v2C_200G(:,6),'--b^','LineWidth',1.0,'MarkerSize',5,'MarkerEdgeColor','blue','MarkerFaceColor','blue');
P3 = plot(MFR,thrust_SC,'k+','LineWidth', LW);
Pn = plot(NaN,NaN,'o', 'Marker', 'none');
xlim([min(MFR) max(MFR)])
ylim([30 100])
xlabel('Anode flow rate (sccm)','Fontsize',font);
yl = ylabel('Thrust (mN)','Fontsize',font);
ylt = yl.Position;
yl.Position = [ylab_pos ylt(2:3)];
legend([P2, P3, Pn, S(2).mainLine, S(2).patch, S(1).patch],'Experimental','Scaling law','','NN, \mu', 'NN, 80% CI', 'NN, 99% CI','Location', 'Northwest','Fontsize',font-6,'NumColumns',2);  
legend('boxoff')
box on;
AX = gca;
AX.PositionConstraint = 'innerposition';
AX.InnerPosition = [0.142738091512805,0.144444440683675,0.762261908487195,0.780555559316325];

annotation('arrow', [0.7540,0.7540], [0.6046,0.4881],'Color','k','LineWidth',1.5);
annotation('ellipse',[0.7303,0.607, 0.0446,0.0619],'Color','k','LineStyle',':','LineWidth',2);

axes('Position',[.665 .26 0.18 0.2]);
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
xticks([66,71,76,81])


saveas(gcf,['Figure_open/Fig_12a_SC_KHT100v2c_MT_V',num2str(Va),'.fig']);
saveas(gcf,['Figure_open/Fig_12a_SC_KHT100v2c_MT_V',num2str(Va),'.png']);
saveas(gcf,['Figure_open/Fig_12a_SC_KHT100v2c_MT_V',num2str(Va),'.svg']);
hold off;
