%Brian O'Malley
%ENME 691 - Industrial AI
%HW3
% Spring 2024

clc;clear;close;

%% Top Matter
format long; format compact;
set(0,'defaultTextInterpreter','latex'); %trying to set the default

sz = 60; %Marker Size
szz = sz/35;
lw = 1;
ms=8;
fs=25;
txtsz = 30;
txtFactor = 0.8;
ax = [0.9,1.4,0.0,2.0];
loc = 'southwest';
pos = [218,114,1478,796];
plotFlag = false;
% txtsz = 24;

%% 1: Set file path

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Specify the location of the libsvm/matlab folder  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dir_lib     = 'F:\NOTES\Classes\Industrial AI\HW3';
cd(dir_lib)

load("FeatMat_train.mat");
load("FeatMat_test.mat");
load("cMatTest.mat");
dir_lib     = 'F:\NOTES\Classes\Industrial AI\HW3\libsvm\matlab';
cd(dir_lib)

%% 2: Import data

%%%%%%%%%%%%%%%%%%%
% Write your code %
%%%%%%%%%%%%%%%%%%%
dirTrainH = "F:\NOTES\Classes\Industrial AI\HW2\HW2\Homework 2\Training\Training\Healthy";
dirTrainF1 = "F:\NOTES\Classes\Industrial AI\HW3\Training\Faulty\Unbalance 1";
dirTrainF2 = "F:\NOTES\Classes\Industrial AI\HW3\Training\Faulty\Unbalance 2";

dirTest = "F:\NOTES\Classes\Industrial AI\HW2\HW2\Homework 2\Testing\Testing";
filesH = dir(dirTrainH +'\*.txt');
filesF1 = dir(dirTrainF1 +'\*.txt');
filesF2 = dir(dirTrainF2 +'\*.txt');
filesT = dir(dirTest +'\*.txt');
%set general use values
haftspeed = 20;% Hz
Fs = 2560;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 38400;             % Length of signal
time = (0:L-1)*T;        % Time vector
FsRange = Fs/L*(0:L-1);
%set the range of interest (1x, 2x, 3x, 4x, harmonics)
low1X = find(FsRange==15);
up1X = find(FsRange==25);
low2X = find(FsRange==35);
up2X = find(FsRange==45);
low3X = find(FsRange==55);
up3X = find(FsRange==65);
low4X = find(FsRange==75);
up4X = find(FsRange==85);
%% 3: Feature extraction / FFT

%%%%%%%%%%%%%%%%%%%
% Write your code %
%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%
%%%healthy data%%%%
%%%%%%%%%%%%%%%%%%%

for i=1:length(filesH)
    tempStr = dirTrainH +"\"+filesH(i).name;
    trainingDataH{i} =readtable(tempStr);
    %get additional data
    t=abs(fft(trainingDataH{i}.Date3_26_2014));
    peakToPeakH(i) = peak2peak(t);
    L1H(i) = norm(t,1);
    L2H(i) = norm(t,2);
    LinfH(i) = norm(t,inf);
    % plot(FsRange,fft(trainingDataH{i}.Date3_26_2014),'o')
    transFFT_H{i} = t; clear t;

end
for i=1:20
    temp = transFFT_H{i}/L*2;
    healthy1X(i) = max(temp(low1X:up1X));
    healthy2X(i) = max(temp(low2X:up2X));
    healthy3X(i) = max(temp(low3X:up3X));
    healthy4X(i) = max(temp(low4X:up4X));
    rmsH(i) = rms(temp);
    stdH(i) = std(temp);

    clear temp;

end

%%%%%%%%%%%%%%%%%%%
%%%faulty 1 data%%%
%%%%%%%%%%%%%%%%%%%

for i=1:length(filesF1)
    tempStr = dirTrainF1 +"\"+filesF1(i).name;
    trainingDataF1{i} =readtable(tempStr);
    t=abs(fft(trainingDataF1{i}.Date3_26_2014));
    %get additional data
    peakToPeakF1(i) = peak2peak(t);
    L1F1(i) = norm(t,1);
    L2F1(i) = norm(t,2);
    LinfF1(i) = norm(t,inf);
    transFFT_F1{i} = t; clear t;

end

for i=1:20
    temp = transFFT_F1{i}/L*2;
    faulty11X(i) = max(temp(low1X:up1X));
    faulty12X(i) = max(temp(low2X:up2X));
    faulty13X(i) = max(temp(low3X:up3X));
    faulty14X(i) = max(temp(low4X:up4X));
    rmsF1(i) = rms(temp);
    stdF1(i) = std(temp);

    clear temp;

end

%%%%%%%%%%%%%%%%%%%
%%%faulty 2 data%%%
%%%%%%%%%%%%%%%%%%%

for i=1:length(filesF2)
    tempStr = dirTrainF2 +"\"+filesF2(i).name;
    trainingDataF2{i} =readtable(tempStr);
    t=abs(fft(trainingDataF2{i}.Date3_26_2014));
    %get additional data
    peakToPeakF2(i) = peak2peak(t);
    L1F2(i) = norm(t,1);
    L2F2(i) = norm(t,2);
    LinfF2(i) = norm(t,inf);
    transFFT_F2{i} = t; clear t;

end
for i=1:20
    temp = transFFT_F2{i}/L*2;
    faulty21X(i) = max(temp(low1X:up1X));
    faulty22X(i) = max(temp(low2X:up2X));
    faulty23X(i) = max(temp(low3X:up3X));
    faulty24X(i) = max(temp(low4X:up4X));
    stdF2(i) = std(temp);
    rmsF2(i) = rms(temp);
    clear temp;

end

%%%%%%%%%%%%%%%%%%%
%%%testing data%%%
%%%%%%%%%%%%%%%%%%%

for i=1:length(filesT)
    tempStr = dirTest +"\"+filesT(i).name;
    testData{i} =readtable(tempStr);
    t=abs(fft(testData{i}.Date3_26_2014));
    %get additional data
    peakToPeakT(i) = peak2peak(t);
    L1T(i) = norm(t,1);
    L2T(i) = norm(t,2);
    LinfT(i) = norm(t,inf);
    transFFT_T{i} = t; clear t;

end

%pick out the peak for the Testing data
for i=1:30
    temp = transFFT_T{i}/L*2;
    testing1X(i) = max(temp(low1X:up1X));
    testing2X(i) = max(temp(low2X:up2X));
    testing3X(i) = max(temp(low3X:up3X));
    testing4X(i) = max(temp(low4X:up4X));
    stdT(i) = std(temp);
    rmsT(i) = rms(temp);
    clear temp;

end
%%%%%%%%%%%%%%%%%%%
%%%group data%%%
%%%%%%%%%%%%%%%%%%%

%all components
tempH = [healthy1X',healthy2X',healthy3X',healthy4X',stdH',rmsH',peakToPeakH',L1H',L2H',LinfH'];
tempF1 = [faulty11X',faulty12X',faulty13X',faulty14X',stdF1',rmsF1',peakToPeakF1',L1F1',L2F1',LinfF1'];
tempF2 = [faulty21X',faulty22X',faulty23X',faulty24X',stdF2',rmsF2',peakToPeakF2',L1F2',L2F2',LinfF2'];
tempT = [testing1X',testing2X',testing3X',testing4X',stdT',rmsT',peakToPeakT',L1T',L2T',LinfT'];

%load components into feature matrix
FeatMat_train = [tempH;tempF1;tempF2];
FeatMat_test = tempT;
cMatTest = [1*ones(10,1);2*ones(10,1);3*ones(10,1)];

[coeff,score,latent,tsquared,explained,mu] = pca(FeatMat_train);
%clear low impact components (only need 1st and 2nd harmonic)
FeatMat_train(:,3:end) =[];
FeatMat_test(:,3:end) = [];
save("FeatMat_train.mat","FeatMat_train");
save("FeatMat_test.mat","FeatMat_test");
save("cMatTest.mat","cMatTest");


%% 4: Plot signals, features

%%%%%%%%%%%%%%%%%%%
% Write your code %
%%%%%%%%%%%%%%%%%%%


%plot time
figure(1)
grid on; hold on; box on; 
axis square;
ax=gca;
ax.FontSize = fs;
% pbaspect([2 1 1])

xlim([0 15]);
ylim([-0.4 0.4]);
yticks([-0.4 -0.2 0 0.2 0.4]);

temp = table2array(trainingDataH{1});
plot(time,temp,'-b');
xlabel('Time [s]','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);
clear temp;

figure(2)
grid on; hold on; box on; 
axis square;
% pbaspect([2 1 1])
ax=gca;
ax.FontSize = fs;
xlim([0 15]);
ylim([-0.4 0.4]);
yticks([-0.4 -0.2 0 0.2 0.4]);
temp = table2array(trainingDataF1{1});
plot(time,temp,'-r');
xlabel('Time [s]','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);
clear temp;

figure(3)
grid on; hold on; box on; 
axis square;
% pbaspect([2 1 1])
ax=gca;
ax.FontSize = fs;
xlim([0 15]);
ylim([-0.4 0.4]);
yticks([-0.4 -0.2 0 0.2 0.4]);
temp = table2array(trainingDataF2{1});
plot(time,temp,'-k');
xlabel('Time [s]','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);
clear temp;

%plot frequency
figure(4)
grid on; hold on; box on;
axis square;
% pbaspect([2 1 1])
ax=gca;
ax.FontSize = fs;
xlim([0 60]);
ylim([0 0.005]);
yticks([ 0 0.001 0.002 0.003 0.004 0.005]);
temp = transFFT_H{1}/L*2;
plot(FsRange, temp,'-b');
xlabel('Frequency [Hz]','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);
clear temp;

figure(5)
grid on; hold on; box on; 
axis square;
% pbaspect([2 1 1])
ax=gca;
ax.FontSize = fs;
xlim([0 60]);
ylim([0 0.005]);
yticks([ 0 0.001 0.002 0.003 0.004 0.005]);
temp = transFFT_F1{1}/L*2;
plot(FsRange, temp,'-r');
xlabel('Frequency [Hz]','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);
clear temp;
% legend('Healthy','Faulty','Location','Northwest','FontSize',fs);
% end

figure(6)
grid on; hold on; box on; 
axis square;
% pbaspect([2 1 1])
ax=gca;
ax.FontSize = fs;
xlim([0 60]);
ylim([0 0.005]);
yticks([ 0 0.001 0.002 0.003 0.004 0.005]);
temp = transFFT_F2{1}/L*2;
plot(FsRange, temp,'-k');
xlabel('Frequency [Hz]','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);
clear temp;
% legend('Healthy','Faulty','Location','Northwest','FontSize',fs);

%plot features
figure(7)
grid on; hold on; box on; 
axis square;
% pbaspect([2 1 1])
ax=gca;
ax.FontSize = fs;
yticks([ 0 0.01 0.02 0.03]);
xlim([0 20]);
ylim([0 0.03]);
plot(healthy1X,'-ob');
plot(faulty11X,'-or');
plot(faulty21X,'-ok');

legend('Healthy','Faulty-1','Faulty-2','Location','Northwest','FontSize',fs);
xlabel('# of Samples','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);

figure(8)
grid on; hold on; box on; 
axis square;
ax=gca;
ax.FontSize = fs;
yticks([ 0 0.01 0.02 0.03]);
% pbaspect([2 1 1])
xlim([0 20]);
ylim([0 0.03]);
plot(healthy2X,'-ob');
plot(faulty12X,'-or');
plot(faulty22X,'-ok');
legend('Healthy','Faulty-1','Faulty-2','Location','Northwest','FontSize',fs);
xlabel('# of Samples','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);

figure(9)
grid on; hold on; box on; 
axis square;
ax=gca;
ax.FontSize = fs;
yticks([ 0 0.01 0.02 0.03]);
% pbaspect([2 1 1])
xlim([0 20]);
ylim([0 0.03]);
plot(healthy3X,'-ob');
plot(faulty13X,'-or');
plot(faulty23X,'-ok');

legend('Healthy','Faulty-1','Faulty-2','Location','Northwest','FontSize',fs);
xlabel('# of Samples','FontSize',fs);
ylabel('Amplitude [-]','FontSize',fs);

figure(10)
grid on; hold on; box on; 
% axis square;
ax=gca;
ax.FontSize = fs;
yticks([ 0 0.01 0.02 0.03]);
pbaspect([2 1 1])

xlim([0 20]);
ylim([0 0.03]);
plot(rmsH,'-ob');
plot(rmsF1,'-or');
plot(rmsF2,'-ok');

legend('Healthy','Faulty-1','Faulty-2','Location','Northwest','FontSize',fs);
xlabel('# of Samples','FontSize',fs);
ylabel('RMS [-]','FontSize',fs);

f11 = figure(11)
grid on; hold on; box on; 
% axis square;
ax=gca;
ax.FontSize = fs;
yticks([ 0 0.01 0.02 0.03]);
pbaspect([2 1 1])
xlim([0 20]);
ylim([0 0.03]);
plot(stdH,'-ob');
plot(stdF1,'-or');
plot(stdF2,'-ok');

legend('Healthy','Faulty-1','Faulty-2','Location','Northwest','FontSize',fs);
xlabel('# of Samples','FontSize',fs);
ylabel('STD [-]','FontSize',fs);

%PCA results
figure(13)
hBar=bar(explained,'b','EdgeColor','k');
set(gca,'xticklabel',{'1X','2X','3X','4X','STD','RMS','P2P','L1','L2','Linf'})
set(gca,'YScale','log')
set(gca,'FontSize',fs)
ylabel('Variance Explained [%]')
%% 5. SVM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Before this section, you need prepare 
%  - FeatMat_train: Feature matrix for training data
%  - FeatMat_test : Feature matrix for testing data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_train = 60;
Nh = 20;
Nf1 = 20;
% Training data
Train_X = FeatMat_train;
Train_Y = zeros(N_train,1);
Train_Y(1:Nh,1) = 1;
Train_Y(Nh+1:Nh+Nf1,1) = 2;
Train_Y(Nh+Nf1+1:N_train,1) = 3;

% Test Data
Test_X = FeatMat_test;
Test_Y = zeros(30,1);
Test_Y( 1:10,1) = 1;
Test_Y(11:20,1) = 2;
Test_Y(21:30,1) = 3;

% train SVM with different kernel

Mehtod_list = {'rbf','linear','polynomial','Sigmoid'}; % kernel function selection

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here you can select kernel function 
% Try different kernel and check the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Method = Mehtod_list{1}; % 1: rbf, 2: linear, 3: polynomial, 4: softmargin

switch Method
    case 'rbf'
            svmStruct = libsvmtrain(Train_Y,Train_X,'-s 0 -t 2 -g 0.333 -c 1');
            % refer to README file in libsvm for more infomation

    case 'linear'
            svmStruct = libsvmtrain(Train_Y,Train_X,'-s 0 -t 0 -g 0.333 ');
        
    case 'polynomial'
            svmStruct = libsvmtrain(Train_Y,Train_X,'-s 0 -t 1 -g 0.333 ');
            
    case 'Sigmoid'
        svmStruct = libsvmtrain(Train_Y,Train_X,'-s 0 -t 3 -g 0.333 ');
        
end

% Test and predict label
% use trained SVM model for classification
[predicted_result, accuracy,~] = libsvmpredict(Test_Y,Test_X,svmStruct);

%% 6. Confusion Matrix

%%%%%%%%%%%%%%%%%%%
% Write your code %
%%%%%%%%%%%%%%%%%%%
figure(12)
cm = confusionchart(cMatTest,predicted_result)
cm.FontSize = fs;


