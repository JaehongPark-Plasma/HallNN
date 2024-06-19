%% reload
clc;clear;
load("Data/DATA_RHT_in.mat");
load("Data/DATA_RHT_out.mat");

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

% NN_minmax: store the min/max of input/output
NN_minmax.input_min = input_min;
NN_minmax.input_max = input_max;
NN_minmax.output_min = output_min;
NN_minmax.output_max = output_max;
% Now, data is --> [0 1] normalized

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
    [NNens{ens_i}, NNTRens{ens_i}] = train(net,input_NN,output_NN,'useParallel','no'); % Donot use parallel training for indiviual NN trains.
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
disp(['### Training process completed :)'])

%% PLOT
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
% Save the plot of trained networks' performances (MSEs)
saveas(gcf,['results/NN_perf','.png']);
