%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Cross-validated Shallow Learning with multilayer perceptron (SL) for different
%%  numbers of neurons in the hidden layer
%%
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [out_kmeans]= ShallowNN_Classify_ts(X,pi, X_valid, pi_valid,flag_AUC)

%% Update for "triple split"
T_v=size(X_valid,2);
T_vh=floor(size(X_valid,2)/2);
X_valid_ts = X_valid(:,(1):T_v);
pi_valid_ts = pi_valid(:,(1):T_v);
X_valid = X_valid(:,1:T_vh);
pi_valid = pi_valid(:,1:T_vh);



N=[2 3 4 5 7 10 15 25 40];
for n=1:length(N)
    tic;
    net =patternnet(N(n));
    net = train(net,X,pi);
    gamma_valid=double(net(X_valid ));
    gamma_valid_ts=double(net(X_valid_ts ));
    out_kmeans.time(n)=toc;
    [L_pred_test] = AUC_of_Prediction(gamma_valid,eye(size(gamma_valid,1)),pi_valid,0,flag_AUC);
    [~,out_kmeans.confusion_matr_valid{n},~,~] = confusion(pi_valid,gamma_valid);
    out_kmeans.L_pred_valid(n)=L_pred_test/(size(pi_valid,2)*size(pi_valid,1));
    [L_pred_test_ts] = AUC_of_Prediction(gamma_valid_ts,eye(size(gamma_valid_ts,1)),pi_valid_ts,0,flag_AUC);
    [~,out_kmeans.confusion_matr_valid_ts{n},~,~] = confusion(pi_valid_ts,gamma_valid_ts);
    out_kmeans.L_pred_valid_ts(n)=L_pred_test_ts/(size(pi_valid_ts,2)*size(pi_valid_ts,1));
    out_kmeans.net{n}=net;
end



