%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Cross-validated Deep Learning with Long-Short Term Memory algorithm (DL LSTM) for diifferent
%%  numbers of neurons in the hidden layer
%%
%%
%% (c) Illia Horenko 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [out_kmeans]= DeepNN_Regression_ts(X,pi, X_valid, pi_valid)



%% Update for "triple split"
T_v=size(X_valid,2);
T_vh=floor(size(X_valid,2)/2);
X_valid_ts = X_valid(:,(1):T_v);
pi_valid_ts = pi_valid(:,(1):T_v);
X_valid = X_valid(:,1:T_vh);
pi_valid = pi_valid(:,1:T_vh);
%%%%%%%%%%%%%%%%%%%%%%%%%%
[d,T]=size(X);T_valid=size(X_valid,2);T_valid_ts=size(X_valid_ts,2);
T_interm=floor(8/10*T);
for t=1:T_interm
    XTrain{t,1}=X(:,t);
    [YTrain{t,1}]=pi(:,t);
end
k=1;
for t=(1+T_interm):T
    XTrain_i{k,1}=X(:,t);
    [YTrain_i{k,1}]=pi(:,t);
    k=k+1;
end
for t=1:T_valid
    XValidation{t,1}=X_valid(:,t);
    YValidation{t,1}=(pi_valid(:,t));
end
%YTrain=categorical(YTrain);
%YTrain_i=categorical(YTrain_i);
%YValidation=categorical(YValidation);

for t=1:T_valid_ts
    XValidation_ts{t,1}=X_valid_ts(:,t);
    [YValidation_ts{t,1}]=(pi_valid_ts(:,t));
end
%YValidation_ts=categorical(YValidation_ts);

numFeatures = d;
numResponses = size(YTrain{1},1);
N=[2 5 10];
%N=[2 4 10 25 50];
%N=[2 3 4 5 7 10 15 25 50 100];
for n=1:length(N)
 %   n
numHiddenUnits = N(n);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(20)
dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',1000, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{XTrain_i,YTrain_i}, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','none');

tic;net = trainNetwork(XTrain,YTrain,layers,options);
for nn=1:numel(XValidation)
gamma_valid(:,nn)=cell2mat(net.predict(XValidation(nn)))';
end
for nn=1:numel(XValidation_ts)
gamma_valid_ts(:,nn)=cell2mat(net.predict(XValidation_ts(nn)))';
end
time(n)=toc;
[L_pred_test(n)] = (1/size(pi_valid,2))*sum(sum((gamma_valid-pi_valid).^2));
%out_kmeans.L_pred_valid(n)=L_pred_test;

[L_pred_test_ts(n)] = (1/size(pi_valid_ts,2))*sum(sum((gamma_valid_ts-pi_valid_ts).^2));
nett{n}=net;
%out_kmeans.N_params(n)=prod(size(net.Layers(2).InputWeights))+...
%    prod(size(net.Layers(2).RecurrentWeights))+prod(size(net.Layers(2).Bias))+prod(size(net.Layers(3).Weights))...
%    +prod(size(net.Layers(3).Bias));
end
[~,n]=min(L_pred_test);
out_kmeans.N_hidden=N(n);
out_kmeans.L_pred_valid_ts=L_pred_test_ts(n);
out_kmeans.time=time(n);
out_kmeans.gamma_valid=[cell2mat(nett{n}.predict(XValidation_ts))'];





