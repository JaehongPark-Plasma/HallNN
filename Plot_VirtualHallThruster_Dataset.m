%% Github plot

%% Figure g1
% Input T
clc;clear;

load('Data/DATA_VHT_in.mat');
load('Data/DATA_VHT_out.mat');
LW = 0.7;
font = 18;
MS = 4.5;

figure(703)
t = tiledlayout(4,5,'TileSpacing','Compact');

nexttile
loglog(rinput(:,1),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,1),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('Anode flow rate (sccm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xticks([1 10 100 1000 10000])
ylim([1e-1 5e4])
yticks([1 100 10000])

% 2
nexttile
loglog(rinput(:,2),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,2),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('V_d (V)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,2))*0.9 max(rinput(:,2))*1.05])
xticks([100 300 500 800])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 3
nexttile
loglog(rinput(:,3),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,3),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('R_{out} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,3))*0.8 max(rinput(:,3))*1.2])
xticks([10 20 50 100 200])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 4
nexttile
loglog(rinput(:,4),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,4),output(:,1),'.','color','#0094FF','linewidth',LW-0.5);
xlabel('R_{in} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,4))*0.8 max(rinput(:,4))*1.05])
xticks([5 20 50 100 200])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 5
nexttile
loglog(rinput(:,5),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,5),output(:,1),'.','color','#0094FF','linewidth',LW-0.5);
xlabel('L_{ch} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,5))*0.9 max(rinput(:,5))*1.05])
xticks([15 20 30 40 50])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 6
nexttile
loglog(rinput(:,6),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,6),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('B_m (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,6))*0.8 max(rinput(:,6))*1.1])
xticks([100 200 400])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 7
nexttile
loglog(rinput(:,7),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,7),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('L_m (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,7))*0.9 max(rinput(:,7))*1.1])
xticks([20 30 40 50])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 8
nexttile
loglog(rinput(:,8),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,8),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('B_1 (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([0.5 max(rinput(:,8))*1.2])
xticks([1 4 10 20 40])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 9
nexttile(11)
loglog(rinput(:,9),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,9),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('c_1 (-)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,9))*0.95 max(rinput(:,9))*1.05])
xticks([1.2 1.5 2 2.5])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 10
nexttile(12)
loglog(rinput(:,10),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,10),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('s_1 (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,10))*0.9 max(rinput(:,10))*1.1])
xticks([2 5 10 15])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 11
nexttile(13)
loglog(rinput(:,11),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,11),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('B_2 (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,11))*0.9 max(rinput(:,11))*1.1])
xticks([5 10 20 50])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 12
nexttile(16)
loglog(rinput(:,12),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,12),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('c_2 (-)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,12))*0.95 max(rinput(:,12))*1.05])
xticks([1.2 1.5 2 2.5])
ylim([1e-1 5e4]); yticks([1 100 10000])

% 13
nexttile(17)
loglog(rinput(:,13),routput(:,1),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,13),output(:,1),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('s_2 (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,13))*0.9 max(rinput(:,13))*1.1])
xticks([5 10 15 20])
ylim([1e-1 5e4]); yticks([1 100 10000])
p1 = plot(NaN,NaN,'o','MarkerSize',4.5,'MarkerFaceColor',[1 0.271 0.0318],'MarkerEdgeColor',[1 0.271 0.0318]);
p2 = plot(NaN,NaN,'o','MarkerSize',4.5,'MarkerFaceColor',[0 0.58 1],'MarkerEdgeColor',[0 0.58 1]);
leg = legend([p1,p2],'Virtual Hall thruster','KAIST Hall thruster','Fontsize',font-2,'location','west');
leg.Layout.Tile = 18;

ylabel(t,'Thrust (mN)','Fontsize',font+6)
title(t,'Input-output relationship in numerical simulation-generated training data','Fontsize',font+6)

x0=50;
y0=50;
width=1600;
height=900;
set(gcf,'position',[x0,y0,width,height])

hold off;
%
saveas(gcf,'results/Dataset_VHT_input_Thrust.png');

% histogram
font = 18;
RanHall_P = rinput(:,2).*routput(:,2)/1e3;
RanHall_T = routput(:,1);
% KHall_P = input(:,2).*output(:,2)/1e3;
% KHall_T = output(:,1);

PP = [RanHall_P;];% KHall_P];
TT = [RanHall_T;];% KHall_T];
RK_name = cell(numel(PP),1);
for i=1:numel(RanHall_P)
RK_name{i} = 'RHT';
end
% for i = numel(RanHall_P)+1:numel(PP)
% RK_name{i} = 'KHT';
% end

% scatter histrogram (sub-plot)
figure(20)
c = scatterhist(PP,TT,'Group',RK_name,'Kernel','on','Bandwidth',[0.5;20],'Location','SouthWest','Direction','in','Color',[0.9961    0.2695    0.3164],'LineStyle',{'-','-'},'LineWidth',1.5,'Marker','o','MarkerSize',4.5,'legend','off');
hold on;
LL = linspace(0,1e5,100);
RH_TP = mean(routput(:,1)./(rinput(:,2).*routput(:,2)))*1e3
%IH_TP = mean(output(:,1)./(input(:,2).*output(:,2)))*1e3
plot(LL,LL*40,'--k','linewidth',LW+2.5);
plot(LL,LL*80,'--k','linewidth',LW+2.5);
xlabel('V_d\cdotI_d (kW)','Fontsize',font)
ylabel('Thrust (mN)','Fontsize',font)
text(0.900,350,'80 mN/kW','Fontsize',font-4)
text(3.500,65,'40 mN/kW','Fontsize',font-4)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-3,'linewidth',LW+0,'Layer','top')
xlim([0 10])
ylim([0 600])
hold off;
x0=50;
y0=50;
width=600;
height=600;
set(gcf,'position',[x0,y0,width,height])

c(1).Children(5).MarkerFaceColor = '#FF4551';
c(1).Children(5).MarkerEdgeColor = '0.2 0.2 0.2';

saveas(gcf,'results/Dataset_VHT_T_Power.png');

%% Figure G2
% Input Id
clc;clear;

load('Data/DATA_VHT_in.mat');
load('Data/DATA_VHT_out.mat');
LW = 0.7;
font = 18;
MS = 4.5;
name = 'Plot_temp\newinput_all_';
figure(703)
t = tiledlayout(4,5,'TileSpacing','Compact');

nexttile
loglog(rinput(:,1),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,1),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('Anode flow rate (sccm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xticks([1 10 100 1000 10000])
ylim([1e-2 1e3])
yticks([1e-1 1e1 1e3])

% 2
nexttile
loglog(rinput(:,2),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,2),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('V_d (V)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,2))*0.9 max(rinput(:,2))*1.05])
xticks([100 300 500 800])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 3
nexttile
loglog(rinput(:,3),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,3),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('R_{out} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,3))*0.8 max(rinput(:,3))*1.2])
xticks([10 20 50 100 200])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 4
nexttile
loglog(rinput(:,4),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,4),output(:,2),'.','color','#0094FF','linewidth',LW-0.5);
xlabel('R_{in} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,4))*0.8 max(rinput(:,4))*1.05])
xticks([5 20 50 100 200])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 5
nexttile
loglog(rinput(:,5),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,5),output(:,2),'.','color','#0094FF','linewidth',LW-0.5);
xlabel('L_{ch} (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,5))*0.9 max(rinput(:,5))*1.05])
xticks([15 20 30 40 50])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 6
nexttile
loglog(rinput(:,6),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,6),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('B_m (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,6))*0.8 max(rinput(:,6))*1.1])
xticks([100 200 400])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 7
nexttile
loglog(rinput(:,7),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,7),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('L_m (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,7))*0.9 max(rinput(:,7))*1.1])
xticks([20 30 40 50])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 8
nexttile
loglog(rinput(:,8),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,8),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('B_1 (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([0.5 max(rinput(:,8))*1.2])
xticks([1 4 10 20 40])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 9
nexttile(11)
loglog(rinput(:,9),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,9),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('c_1 (-)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,9))*0.95 max(rinput(:,9))*1.05])
xticks([1.2 1.5 2 2.5])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 10
nexttile(12)
loglog(rinput(:,10),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,10),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('s_1 (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,10))*0.9 max(rinput(:,10))*1.1])
xticks([2 5 10 15])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 11
nexttile(13)
loglog(rinput(:,11),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,11),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('B_2 (G)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,11))*0.9 max(rinput(:,11))*1.1])
xticks([5 10 20 50])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 12
nexttile(16)
loglog(rinput(:,12),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,12),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('c_2 (-)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,12))*0.95 max(rinput(:,12))*1.05])
xticks([1.2 1.5 2 2.5])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])

% 13
nexttile(17)
loglog(rinput(:,13),routput(:,2),'.','color','#FF4551','Markersize',MS,'linewidth',LW-0.5);
hold on
%plot(input(:,13),output(:,2),'.','color','#0094FF','Markersize',MS,'linewidth',LW-0.5);
xlabel('s_2 (mm)','Fontsize',font)
set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-5,'linewidth',LW-0.5,'Layer','top')
xlim([min(rinput(:,13))*0.9 max(rinput(:,13))*1.1])
xticks([5 10 15 20])
ylim([1e-2 1e3]); yticks([1e-1 1e1 1e3])
p1 = plot(NaN,NaN,'o','MarkerSize',4.5,'MarkerFaceColor',[1 0.271 0.0318],'MarkerEdgeColor',[1 0.271 0.0318]);
p2 = plot(NaN,NaN,'o','MarkerSize',4.5,'MarkerFaceColor',[0 0.58 1],'MarkerEdgeColor',[0 0.58 1]);
%leg = legend([p1,p2],'Virtual Hall thruster','KAIST Hall thruster','Fontsize',font-2,'location','west');
leg = legend([p1],'Virtual Hall thruster','Fontsize',font-2,'location','west');

leg.Layout.Tile = 18;

ylabel(t,'Discharge current (A)','Fontsize',font+6)
title(t,'Input-output relationship in numerical simulation-generated training data','Fontsize',font+6)

x0=50;
y0=50;
width=1600;
height=900;
set(gcf,'position',[x0,y0,width,height])

hold off;
%
saveas(gcf,'results/Dataset_VHT_input_Id.png');

% histogram
font = 18;
RanHall_P = rinput(:,2).*routput(:,2)/1e3;
RanHall_T = routput(:,1);

PP = [RanHall_P;];% KHall_P];
TT = [RanHall_T;];% KHall_T];
RK_name = cell(numel(PP),1);
for i=1:numel(RanHall_P)
RK_name{i} = 'RHT';
end
for i = numel(RanHall_P)+1:numel(PP)
RK_name{i} = 'KHT';
end
% scatter histrogram (sub-plot)
figure(20)
c = scatter(RanHall_P,RanHall_T,15,'MarkerFaceColor',[0.9961    0.2695    0.3164], MarkerEdgeColor = '0.2 0.2 0.2');
hold on;

dd = rinput(:,2).*routput(:,2);
LL = linspace(0,1e5,100);
RH_TP = mean(routput(:,1)./(rinput(:,2).*routput(:,2)))*1e3

xlabel('V_d\cdotI_d (kW)','Fontsize',font)
ylabel('Thrust (mN)','Fontsize',font)

set(gca,'XMinorTick','on','YMinorTick','on','Fontsize',font-3,'linewidth',LW+0,'Layer','top')
grid on; box on;
set(gca, 'XScale', 'log', 'YScale', 'log')
xlim([1e-4 1000])
ylim([1e-1 2e4])
hold off;
x0=50;
y0=50;
width=600;
height=600;
set(gcf,'position',[x0,y0,width,height])

saveas(gcf,'results/Dataset_VHT_T_Power_loglog.png');
