%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [out] = LogitRegressionCross_v2_ACC(X_train,Y_train,X_valid,Y_valid)
Lambda=10.^[-8:1:1];
T_v=size(X_valid,2);
T_vh=floor(T_v/2);
X_valid_ts=X_valid(:,(1):T_v);
Y_valid_ts=Y_valid(:,(1):T_v);
X_valid=X_valid(:,(1):T_vh);
Y_valid=Y_valid(:,(1):T_vh);
for i=1:length(Lambda)
    tic;
    [B,FitInfo] = lassoglm(X_train',logical(Y_train(1,:))','binomial','Lambda',Lambda(i),'MaxIter',1e3);
    B0 = FitInfo.Intercept;
    coef = [B0; B];
    yhat = glmval(coef,X_valid','logit');
    score_log = yhat;%(yhat>=0.5);
    S=sum(Y_valid(1,:)==round(score_log)')./length(double(score_log)');
    %% Addition for "triple split"
    yhat_ts = glmval(coef,X_valid_ts','logit');
    score_log = yhat_ts;%(yhat>=0.5);
    S_ts=sum(Y_valid_ts(1,:)==round(score_log)')./length(double(score_log)');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tt=toc;
    if i==1
        out.L_pred_valid_v=S;
        %% Addition for "triple split"
        out.L_pred_valid=S_ts;
        %%%%%%%%%%%%%%%%%%%%%%%%
        out.coef=coef;
        out.time=tt;
    else
        if out.L_pred_valid_v<S
            out.L_pred_valid_v=S;
            %% Addition for "triple split"
            out.L_pred_valid=S_ts;
            %%%%%%%%%%%%%%%%%%%%%%%%
            out.coef=coef;
            out.time=tt;
        end
    end
end

