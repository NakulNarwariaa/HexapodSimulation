clear;close all;clc; 

% Forward kinematics plotting

px= -1:0.3:1;
py=-1:0.3:1;
pz=5:0.8:10;
alpha= -30:10:30;
beta= -30:10:30;
gamma=  -30:10:30;


a=10;
b=15;
d=1;

% converting degrees to radians5
alpha=deg2rad(alpha);
beta=deg2rad(beta);
gamma=deg2rad(gamma);

display(alpha)

% Defining the size of l1 to l6
l1=zeros([1 (size(alpha,2)*size(beta,2)*size(gamma,2)*size(px,2)*size(py,2)*size(pz,2))]);
l2=l1;
l3=l1;
l4=l1;
l5=l1;
l6=l1;

% Defining the size of i1 to i6
i1=zeros([1 (size(alpha,2)*size(beta,2)*size(gamma,2)*size(px,2)*size(py,2)*size(pz,2))]);
i2=i1;
i3=i1;
i4=i1;
i5=i1;
i6=i1;

r=1;

% Equation for l1-l6

% SIMULATION DATA GENERATION - Making combinations of positions and orientations
for i =1: size(px,2)
    for j = 1:size(py,2) 
      for k=1:size(pz,2)
        for l=1:size(alpha,2) 
          for m=1:size(beta,2)
            for n=1:size(gamma,2)
               
               xt1= px(i) + a*(sin(alpha(l))*sin(beta(m))*sin(gamma(n)+deg2rad(60))+cos(beta(m))*cos(gamma(n)+deg2rad(60)))/sqrt(3);
               yt1 = py(j) + a*(cos(alpha(l))*sin(gamma(n)+deg2rad(60)))/sqrt(3);
               zt1 = pz(k) + a*(sin(alpha(l)) *cos(beta(m)) * sin(gamma(n)+deg2rad(60)) - sin(beta(m))*cos(gamma(n)+deg2rad(60)))/sqrt(3);
               xt2 = px(i) - a*(sin(alpha(l))*sin(beta(m))*sin(gamma(n)) + cos(beta(m))*cos(gamma(n)))/sqrt(3);
               yt2 = py(j) - a*(cos(alpha(l))*sin(gamma(n)))/sqrt(3);
               zt2 = pz(k)- a*(sin(alpha(l)) * cos(beta(m)) *sin(gamma(n)) - sin(beta(m))*cos(gamma(n)))/sqrt(3);
               xt3 = px(i) + a*(sin(alpha(l))*sin(beta(m))*sin(gamma(n)-deg2rad(60)) + cos(beta(m))*cos(gamma(n)-deg2rad(60)))/sqrt(3);
               yt3 = py(j) + a*(cos(alpha(l))*sin(gamma(n)-deg2rad(60)))/sqrt(3);      
               zt3 = pz(k) + a*(sin(alpha(l))*cos(beta(m))*sin(gamma(n)-deg2rad(60)) - sin(beta(m))*cos(gamma(n)-deg2rad(60)))/sqrt(3);                

               l1(r) = sqrt((xt1-d/(2*sqrt(3))-b/sqrt(3))*(xt1-d/(2*sqrt(3))-b/sqrt(3)) + (yt1-d/2)*(yt1-d/2)+zt1*zt1);
               l2(r) = sqrt((xt1-d/(2*sqrt(3))+b/(2*sqrt(3)))*(xt1-d/(2*sqrt(3))+b/(2*sqrt(3))) + (yt1-d/2-b/2)*(yt1-d/2-b/2) + zt1*zt1);
               l3(r) = sqrt((xt2+d/sqrt(3)+b/(2*sqrt(3)))*(xt2+d/sqrt(3)+b/(2*sqrt(3))) + (yt2-b/2)*(yt2-b/2) + zt2*zt2);
               l4(r) = sqrt((xt2+d/sqrt(3)+b/(2*sqrt(3)))*(xt2+d/sqrt(3)+b/(2*sqrt(3))) + (yt2+b/2)*(yt2+b/2) + zt2*zt2);
               l5(r) = sqrt((xt3-d/(2*sqrt(3))+b/(2*sqrt(3)))*(xt3-d/(2*sqrt(3))+b/(2*sqrt(3))) + (yt3+b/2+d/2)*(yt3+b/2+d/2)+zt3*zt3);
               l6(r) = sqrt((xt3-d/(2*sqrt(3))-b/sqrt(3))*(xt3-d/(2*sqrt(3))-b/sqrt(3)) + (yt3+d/2)*(yt3+d/2)+zt3*zt3);
     
              i1(r)=px(i);
              i2(r)=py(j);
              tmp  = 2*(pz(k)-5)/5 - 1;
              i3(r)=tmp;  
              i4(r)=alpha(l);
              i5(r)=beta(m);
              i6(r) = gamma(n);      

              r=r+1;
            end
          end
        end
      end
    end
end

% k-mean clustering into 4 classes.
l1=l1(:);
l2=l2(:);
l3=l3(:);
l4=l4(:);
l5=l5(:);
l6=l6(:);
i1=i1(:);
i2=i2(:);
i3=i3(:);
i4=i4(:);
i5=i5(:);
i6=i6(:);
lengths = [l1,l2,l3,l4,l5,l6];
in1=[i1,i2,i3,i4,i5,i6];
[idx2,C2]=kmeans(in1,4);

% Target 1--> Class 1
position11=in1(idx2==1,1);
position12=in1(idx2==1,2);
position13=in1(idx2==1,3);
theta14=in1(idx2==1,4);
theta15=in1(idx2==1,5);
theta16=in1(idx2==1,6);
Target1=[position11,position12,position13,theta14,theta15,theta16];

%input 1--> Class 1
l1class1 = lengths(idx2==1,1);
l2class1 = lengths(idx2==1,2);
l3class1 = lengths(idx2==1,3);
l4class1 = lengths(idx2==1,4);
l5class1 = lengths(idx2==1,5);
l6class1 = lengths(idx2==1,6);
input11 = [l1class1,l2class1,l3class1,l4class1,l5class1,l6class1];
input1=input11';
target1=Target1';

trainFcn = 'trainbr';  % Bayesian Regularization
    
    % Create a Feedforward Network
    hiddenLayerSize = [10,10,10,10,10];
    net1 = feedforwardnet (hiddenLayerSize,trainFcn);
    net1.divideFcn = 'divideblock'; % Divide targets into three sets using blocks of indices
    net1.divideParam.trainRatio = 70/100;
    net1.divideParam.valRatio = 15/100;
    net1.divideParam.testRatio = 15/100;
    
    %TRAINING PARAMETERS
    net1.trainParam.show=50;  %# of ephocs in display
    net1.trainParam.lr=0.05;  %learning rate
    net1.trainParam.epochs=200;  %max epochs
    net1.performFcn='mse';  %Name of a network performance function %type help nnperformance
   

    % Train the Network
    [net1,tr] = train(net1,input1,target1);
    % Test the Network
    output1 = net1(input1);
    performance1 = perform(net1,target1,output1);


% Target 2--> Class 2
position21=in1(idx2==2,1);
position22=in1(idx2==2,2);
position23=in1(idx2==2,3);
theta24=in1(idx2==2,4);
theta25=in1(idx2==2,5);
theta26=in1(idx2==2,6);
Target2=[position21,position22,position23,theta24,theta25,theta26];

%input 1--> Class 2
l1class2 = lengths(idx2==2,1);
l2class2 = lengths(idx2==2,2);
l3class2 = lengths(idx2==2,3);
l4class2 = lengths(idx2==2,4);
l5class2 = lengths(idx2==2,5);
l6class2 = lengths(idx2==2,6);
input2 = [l1class2,l2class2,l3class2,l4class2,l5class2,l6class2];
%display(Target1);
input2=input2';
target2=Target2';


trainFcn = 'trainbr';  % Bayesian Regularization
    
    % Create a Feedforward Network
    hiddenLayerSize = [10,10,10,10,10];
    net2 = feedforwardnet (hiddenLayerSize,trainFcn);
    net2.divideFcn = 'divideblock'; % Divide targets into three sets using blocks of indices
    net2.divideParam.trainRatio = 70/100;
    net2.divideParam.valRatio = 15/100;
    net2.divideParam.testRatio = 15/100;
    
    %TRAINING PARAMETERS
    net2.trainParam.show=50;  %# of ephocs in display
    net2.trainParam.lr=0.05;  %learning rate
    net2.trainParam.epochs=200;  %max epochs
    net2.performFcn='mse';  %Name of a network performance function %type help nnperformance
   

    % Train the Network
    [net2,tr] = train(net2,input2,target2);
    % Test the Network
    output2 = net2(input2);
    performance2 = perform(net2,target2,output2);

% Target 3--> Class 3
position31=in1(idx2==3,1);
position32=in1(idx2==3,2);
position33=in1(idx2==3,3);
theta34=in1(idx2==3,4);
theta35=in1(idx2==3,5);
theta36=in1(idx2==3,6);
Target3=[position31,position32,position33,theta34,theta35,theta36];

%input 3--> Class 3
l1class3 = lengths(idx2==3,1);
l2class3 = lengths(idx2==3,2);
l3class3 = lengths(idx2==3,3);
l4class3 = lengths(idx2==3,4);
l5class3 = lengths(idx2==3,5);
l6class3 = lengths(idx2==3,6);
input3 = [l1class3,l2class3,l3class3,l4class3,l5class3,l6class3];
%display(Target1);
input3=input3';
target3=Target3';


trainFcn = 'trainbr';  % Bayesian Regularization
    
    % Create a Feedforward Network
    hiddenLayerSize = [10,10,10,10,10];
    net3 = feedforwardnet (hiddenLayerSize,trainFcn);
    net3.divideFcn = 'divideblock'; % Divide targets into three sets using blocks of indices
    net3.divideParam.trainRatio = 70/100;
    net3.divideParam.valRatio = 15/100;
    net3.divideParam.testRatio = 15/100;
    
    %TRAINING PARAMETERS
    net3.trainParam.show=50;  %# of ephocs in display
    net3.trainParam.lr=0.05;  %learning rate
    net3.trainParam.epochs=200;  %max epochs
    net3.performFcn='mse';  %Name of a network performance function %type help nnperformance
  

    
    % Train the Network
    [net3,tr] = train(net3,input3,target3);
    % Test the Network
    output3 = net3(input3);
    performance3 = perform(net3,target3,output3);

    
% Target 4--> Class 4
position41=in1(idx2==4,1);
position42=in1(idx2==4,2);
position43=in1(idx2==4,3);
theta44=in1(idx2==4,4);
theta45=in1(idx2==4,5);
theta46=in1(idx2==4,6);
Target4=[position41,position42,position43,theta44,theta45,theta46];

% input 4--> Class 4
l1class4 = lengths(idx2==4,1);
l2class4 = lengths(idx2==4,2);
l3class4 = lengths(idx2==4,3);
l4class4 = lengths(idx2==4,4);
l5class4 = lengths(idx2==4,5);
l6class4 = lengths(idx2==4,6);
input4 = [l1class4,l2class4,l3class4,l4class4,l5class4,l6class4];
input4=input4';
target4=Target4';


trainFcn = 'trainbr';  % Bayesian Regularization
    
%     Create a Feedforward Network
    hiddenLayerSize = [10,10,10,10,10];
    net4 = feedforwardnet (hiddenLayerSize,trainFcn);
    net4.divideFcn = 'divideblock'; % Divide targets into three sets using blocks of indices
    net4.divideParam.trainRatio = 70/100;
    net4.divideParam.valRatio = 15/100;
    net4.divideParam.testRatio = 15/100;
  
    
%     TRAINING PARAMETERS
    net4.trainParam.show=50;  %# of ephocs in display
    net4.trainParam.lr=0.05;  %learning rate
    net4.trainParam.epochs=200;  %max epochs
    net4.performFcn='mse';  %Name of a network performance function %type help nnperformance
    
%     Train the Network
    [net4,tr] = train(net4,input4,target4);
%     Test the Network
    output4 = net4(input4);
    performance4 = perform(net4,target4,output4);

    
    
%Calculating Mean squared error in 100 random values    
pposerror1 = 0;
pthetaerror1 = 0;
pposerror2 = 0;
pthetaerror2 = 0;
pposerror3 = 0;
pthetaerror3 = 0;
pposerror4 = 0;
pthetaerror4 = 0;
    for i =1: 100
    
          plengths = [lengths(i*100,1),lengths(i*100,2),lengths(i*100,3),lengths(i*100,4),lengths(i*100,5),lengths(i*100,6)];plengths = plengths';
pin=[in1(i*100,1),in1(i*100,2),in1(i*100,3),in1(i*100,4),in1(i*100,5),in1(i*100,6)];pin=pin';
    poutput1 = net1(plengths);
    tmp = (poutput1(3) + 1)*5/2 + 5;
    poutput1(3)=tmp;
    tmp2 = (pin(3) + 1)*5/2 + 5;
    pin(3)=tmp2;
    pposerror1 =pposerror1 + ((poutput1(1)-pin(1))^2 +(poutput1(2)-pin(2))^2+(poutput1(3)-pin(3))^2);
    pthetaerror1 = pthetaerror1 + ((poutput1(4)-pin(4))^2+(poutput1(5)-pin(5))^2+(poutput1(6)-pin(6))^2);
    
    
    poutput2 = net2(plengths);
     tmp = (poutput2(3) + 1)*5/2 + 5;
    poutput2(3)=tmp;
    pposerror2 =pposerror2 +((poutput2(1)-pin(1))^2 +(poutput2(2)-pin(2))^2+(poutput2(3)-pin(3))^2);
    pthetaerror2 = pthetaerror2 + ((poutput2(4)-pin(4))^2+(poutput2(5)-pin(5))^2+(poutput2(6)-pin(6))^2);
     
    poutput3 = net3(plengths);
     tmp = (poutput3(3) + 1)*5/2 + 5;
    poutput3(3)=tmp;


   pposerror3 =pposerror3 + ((poutput3(1)-pin(1))^2 +(poutput3(2)-pin(2))^2+(poutput3(3)-pin(3))^2);
    pthetaerror3 = pthetaerror3 + ((poutput3(4)-pin(4))^2+(poutput3(5)-pin(5))^2+(poutput3(6)-pin(6))^2);
     
    poutput4 = net4(plengths);
     tmp = (poutput4(3) + 1)*5/2 + 5;
    poutput4(3)=tmp;
   poutput4(6) = (poutput4(6) + 1)*5/2 + 5;

   pposerror4 =pposerror4 + ((poutput4(1)-pin(1))^2 +(poutput4(2)-pin(2))^2+(poutput4(3)-pin(3))^2);
    pthetaerror4 = pthetaerror4 + ((poutput4(4)-pin(4))^2+(poutput4(5)-pin(5))^2+(poutput4(6)-pin(6))^2);
    [pin,poutput4]
    
    end   
    
    pposerror1 = pposerror1/100;
    pthetaerror1 = pthetaerror1/100;
    pposerror2 = pposerror2/100;
    pthetaerror2 = pthetaerror2/100;
    pposerror3 = pposerror3/100;
    pthetaerror3 = pthetaerror3/100;
    pposerror4 = pposerror4/100;
    pthetaerror4 = pthetaerror4/100;
    
    pposerror1
    pthetaerror1
    pposerror2
    pthetaerror2
    pposerror3
    pthetaerror3
    pposerror4
    pthetaerror4
