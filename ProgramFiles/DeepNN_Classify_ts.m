%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Cross-validated Deep Learning with Long-Short Term Memory algorithm (DL LSTM) for diifferent
%%  numbers of neurons in the hidden layer
%%
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [out_kmeans]= DeepNN_Classify_ts(X,pi, X_valid, pi_valid,flag_AUC)



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
    [~,YTrain(t,1)]=max(pi(:,t));
end
k=1;
for t=(1+T_interm):T
    XTrain_i{k,1}=X(:,t);
    [~,YTrain_i(k,1)]=max(pi(:,t));
    k=k+1;
end
for t=1:T_valid
    XValidation{t,1}=X_valid(:,t);
    [~,YValidation(t,1)]=max(pi_valid(:,t));
end
YTrain=categorical(YTrain);
YTrain_i=categorical(YTrain_i);
YValidation=categorical(YValidation);

for t=1:T_valid_ts
    XValidation_ts{t,1}=X_valid_ts(:,t);
    [~,YValidation_ts(t,1)]=max(pi_valid_ts(:,t));
end
YValidation_ts=categorical(YValidation_ts);

numFeatures = d;
numClasses = size(pi,1);
N=[2 3 4 5 7 10 15 25];
for n=1:length(N)
 %   n
numHiddenUnits = N(n);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
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
gamma_valid=double(net.predict(XValidation))';
gamma_valid_ts=double(net.predict(XValidation_ts))';
out_kmeans.time(n)=toc;
[L_pred_test] = AUC_of_Prediction(gamma_valid,eye(size(pi_valid,1)),pi_valid,0,flag_AUC);
out_kmeans.L_pred_valid(n)=L_pred_test/(size(pi_valid,2)*size(pi_valid,1));
[~,out_kmeans.confusion_matr_valid{n},~,~] = confusion(pi_valid,gamma_valid);

[L_pred_test_ts] = AUC_of_Prediction(gamma_valid_ts,eye(size(pi_valid_ts,1)),pi_valid_ts,0,flag_AUC);
out_kmeans.L_pred_valid_ts(n)=L_pred_test_ts/(size(pi_valid_ts,2)*size(pi_valid_ts,1));
[~,out_kmeans.confusion_matr_valid_ts{n},~,~] = confusion(pi_valid_ts,gamma_valid_ts);
out_kmeans.net{n}=net;
out_kmeans.N_params(n)=prod(size(net.Layers(2).InputWeights))+...
    prod(size(net.Layers(2).RecurrentWeights))+prod(size(net.Layers(2).Bias))+prod(size(net.Layers(3).Weights))...
    +prod(size(net.Layers(3).Bias));
end



