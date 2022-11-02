%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Cross-validated Shallow Learning with multilayer perceptron (SL) for different
%%  numbers of neurons in the hidden layer
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [out]= ShallowNN_Regression_ts(X,pi, X_valid, pi_valid)

%% Update for "triple split"
T_v=size(X_valid,2);
T_vh=floor(size(X_valid,2)/2);
X_valid_ts = X_valid(:,(1):T_v);
pi_valid_ts = pi_valid(:,(1):T_v);
X_valid = X_valid(:,1:T_vh);
pi_valid = pi_valid(:,1:T_vh);



N=[2 5 10];%[2 3 4 5 7 10 15 25 40];
for n=1:length(N)
    tic;
    net =feedforwardnet(N(n));
    net.trainParam.showWindow=0;
    net = train(net,X,pi);
    time(n)=toc;
    gamma_train{n}=double(net(X));
    gamma_valid{n}=double(net(X_valid ));
    gamma_test{n}=double(net(X_valid_ts ));
    L_train(n)=Pred_Error_L2(pi,gamma_train{n}) ;
    L_valid(n)=Pred_Error_L2(pi_valid,gamma_valid{n}) ;
    L_test(n)=Pred_Error_L2(pi_valid_ts,gamma_test{n}) ;
end

[~,e]=min(L_valid);
out.L_valid=L_valid(e);
out.L_train=L_train(e);
out.L_test=L_test(e);
out.N_hidden=N(e);
out.gamma_test=gamma_test{e};
out.gamma_valid=gamma_valid{e};
out.gamma_train=gamma_train{e};
out.time=time(e);
end

function y=Pred_Error_L2(y_real,y_pred) 
y=sum(sum((y_real-y_pred).^2))./size(y_real,2);
end