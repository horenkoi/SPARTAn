%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Comparison of regression learners on historical ENSO predictions
%% 
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all
close all
addpath('ProgramFiles/')
addpath('ProgramFiles/tSNE_matlab/')
addpath('Input/')
load('enso_predict_12.mat');
bins=[0 0.2:0.1:0.8 1];
month_lag=5;
enso_lag =8;
pi=X(1,(enso_lag+month_lag):(size(X,2)-1));
xx=[];
for i=1:enso_lag
    xx=[xx;X(1,i:(size(X,2)-enso_lag-month_lag+i-1))];
end
%X=[xx(enso_lag,:);diff(xx);X(2:101,i:(size(X,2)-enso_lag-month_lag+i-1))];
X=[xx(enso_lag,:);diff(xx);X(:,i:(size(X,2)-enso_lag-month_lag+i-1))];
%pi=pi';
%[p,S] = polyfit(1:length(pi),pi,1);
%[y_fit,delta] = polyval(p,1:length(pi),S);
%pi=pi-y_fit;
%pi=abs(pi');
X=X';
%% Set the number N of feature diimensions and T data instances
for d=1:size(X,2);
    X(:,d)=X(:,d)-min(X(:,d));
    if max(abs(X(:,d)))>0
    X(:,d)=X(:,d)./max(abs(X(:,d)));
    end
end
% xx=X;
% %% Please uncomment the code below if you want to extend features with their nonlinear product combinations 
% for d1=1:size(xx,2) 
%     %for d2=(d1):size(xx,2)
%         X=[X sqrt(xx(:,d1))];
%     %end
% end
%X=xx;

% for d1=1:size(xx,2) 
%     for d2=(d1):size(xx,2)
%         X=[X (xx(:,d1)).*(xx(:,d2))];
%     end
% end
% 


X=X';
pi=(pi)';
for j=1:size(pi,2);
    min_pi(j)=min(pi(:,j));
    pi(:,j)=pi(:,j)-min(pi(:,j));
    max_pi(j)=max(abs(pi(:,j)));
    if max(abs(pi(:,j)))>0
    pi(:,j)=pi(:,j)./max(abs(pi(:,j)));
    end
end


pi=pi';

rand('seed',1);
randn('seed',1);


%% Set this flag to 1 if you have the licence for a "Parallel Computing" toolbox of MATLAB
flag_parallel=1;
flag_persistent_prediction=1;
%% Number of SPARTA patterns/boxes/clusters
K=3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set the grid for SPARTA regularisation parameter in front of the entropy-regularisation term
reg_param_W_SPARTA=[1e-4 5e-4 1e-3 2e-3 4e-3 6e-3 8e-3 1e-2 0.05 10];%[1e-9 1e-8 1e-7 1e-6 1e-5 5e-5 0.0001 0.0004 0.0008 1e-3 5e-3 1e-2 5e-2 0.1];%[0.0001 0.0003 0.0005 0.0008 0.001 0.002 0.005];% [5e-9 1e-8 5e-8 1e-7 1e-6 5e-6 1e-5  2e-5 5e-5 1e-4 2e-4 3e-4 4e-4 7e-4 1e-3 5e-3];
%% Set the grid for SPARTA reguularisation parameter in front of the regression
reg_param_CL_SPARTA=[0.1 5 30];%[ 1e-5 1e-4  5e-4 1e-3 5e-3 1e-2 1e-1 1 10 100 500];
%% Set the grid for SPARTA regularisation parameter in front of the l2-regularisation term
reg_param_L_SPARTA=[5e-5 5e-4 1-4];%[1e-7 1e-5 1e-4];%[1e-9 1e-7 1e-5 1e-4];


T=size(X,2);
%% Number of crossvalidation steps 
N_anneal=10;
%% Please keep the valuue flag_AUC=0, implying prediction accuracy as a performance metrics
flag_AUC=0;
%% Defines a fraction of data used for training (i.e. fraction=0.5 means that 50% of data is used for training and another 50% for model validation and testing)
fraction_int=[0.7 0.8];

RF_AUC=zeros(1,N_anneal);
RF_AUC_train=zeros(1,N_anneal);

k=1;
for i=1:length(reg_param_W_SPARTA)
    for j=1:length(reg_param_CL_SPARTA)
        for l=1:length(reg_param_L_SPARTA)
            reg_param(:,k)=[reg_param_W_SPARTA(i);reg_param_CL_SPARTA(j);reg_param_L_SPARTA(l)];
            k=k+1;
        end
    end
end
paroptions = statset('UseParallel',true);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i_prop=1;
for i_ann=1:N_anneal
    %% Random permuutation of data and splitting into training and validation pluus test sets
    %ind_perm=randperm(T);
    ind_perm=1:T;

    if length(fraction_int>1)
        fraction=(fraction_int(2)-fraction_int(1))*rand(1)+fraction_int(1);
    else
        fraction=fraction_int;
    end
    [~,ind_back]=sort(ind_perm,'ascend');
    X=X(:,ind_perm);
    pi=pi(:,ind_perm);
    X_train=X(:,1:floor(fraction*T));pi_train=pi(:,1:floor(fraction*T));
    X_valid=X(:,(1+floor(fraction*T)):T);pi_valid=pi(:,(1+floor(fraction*T)):T);
    %% Run the SPARTA algorithm
%     out_SPARTA_v2 = SPARTA_v2(X_train,pi_train,...
%         K,6,[],reg_param,X_valid,pi_valid,flag_parallel)
%     t_SPARTA_v2(i_prop,i_ann)=out_SPARTA_v2.time_SPARTA
%     SPARTA_AUC_v2(i_prop,i_ann)=out_SPARTA_v2.L_test

    out_SPARTA = SPARTA(X_train,pi_train,...
        K,5,[],reg_param,X_valid,pi_valid,flag_parallel)
    % [out_opd] = SPARTA(X,Pi,K,N_anneal,out_init,reg_param,X_valid,pi_valid,flag_parallel)

    SPARTA_AUC(i_prop,i_ann)=out_SPARTA.L_test
    W_SPARTA{i_ann}=out_SPARTA.W; 
    t_SPARTA(i_prop,i_ann)=out_SPARTA.time_SPARTA
    eps_W(i_prop,i_ann)=out_SPARTA.reg_param_W;
    eps_CL(i_prop,i_ann)=out_SPARTA.reg_param_CL;
    eps_L2(i_prop,i_ann)=out_SPARTA.reg_param_L2;
    [SPARTA_Bin(i_ann,:),x] = MSE_bins(pi_valid,(pi_valid-out_SPARTA.gamma_test).^2,bins);

    clear Lambda
    Lambda=SPARTA_output_Lambda(out_SPARTA,1);
    Lambda=Lambda.^2;D=size(Lambda,2);
    for k=1:size(Lambda,1)
        Lambda(k,2:D)= Lambda(k,2:D)./sum(sum(Lambda(k,2:D)));
    end
    if size(Lambda,1)>1 
        ml(i_ann,:)=mean(Lambda(:,2:D));
    else
        ml(i_ann,:)=(Lambda(:,2:D));  
    end


    %% Ruun Regression Tree Ensemble
    tic;
    t = templateTree('MaxNumSplits',1);
    clear y_predict y_train
    for j=1:size(pi,1)
    Mdl = fitrensemble(X_train',pi_train(j,:));
    y_predict(j,:)=predict(Mdl,X_valid')';
    y_train(j,:)=predict(Mdl,X_train')';
    end
    RF_AUC(i_prop,i_ann)=sum(sum((y_predict-pi_valid).^2))./size(X_valid,2)
    RF_AUC_train(i_prop,i_ann)=sum(sum((y_train-pi_train).^2))./size(X_train,2);
    %end
    t_RF(i_prop,i_ann)=toc;
    RF_Bin(i_ann,:) = MSE_bins(pi_valid,(pi_valid-y_predict).^2,bins);
        %% Run the shallow NN algorithm 
    out_SN= ShallowNN_Regression_ts(X_train,pi_train, X_valid, pi_valid);
     t_SN(i_prop,i_ann)=out_SN.time;
    SN_AUC(i_prop,i_ann)=out_SN.L_test
    SN_Bin(i_ann,:) = MSE_bins(pi_valid,(pi_valid-out_SN.gamma_test).^2,bins);


    tic;
    mean_pi=mean(pi_train);
    MP_AUC(i_prop,i_ann)=sum(sum((mean_pi-pi_valid(j,:)).^2))./size(pi_valid,2)
    t_MP(i_prop,i_ann)=toc;
    MP_Bin(i_ann,:) = MSE_bins(pi_valid,(pi_valid-mean_pi).^2,bins);

    if flag_persistent_prediction==1;
    tic;
    mean_pi=mean(pi_train);
    PP_AUC(i_prop,i_ann)=sum(sum((X_valid(1,:)-pi_valid(j,:)).^2))./size(pi_valid,2)
    t_PP(i_prop,i_ann)=toc;
    PP_Bin(i_ann,:) = MSE_bins(pi_valid,(pi_valid-X_valid(1,:)).^2,bins);
    end
    out= lasso_cross(X_train,pi_train, X_valid, pi_valid,10.^(-9:2),10.^(-9:3)); 
    Lin_AUC(i_prop,i_ann)=out.min_par_err
    t_Lin(i_prop,i_ann)=out.time;
    Lin_Bin(i_ann,:) = MSE_bins(pi_valid,(pi_valid-out.y_valid).^2,bins);

    [out_DL]= DeepNN_Regression_ts(X_train,pi_train, X_valid, pi_valid);
    DL_AUC(i_prop,i_ann)=out_DL.L_pred_valid_ts
    t_DL(i_prop,i_ann)=out_DL.time;
    DL_Bin(i_ann,:) = MSE_bins(pi_valid,(pi_valid-out_DL.gamma_valid).^2,bins);

% figure;subplot(2,3,2);plot(pi_train(1,:),pi_train(2,:),'.');hold on;plot(out_SN.gamma_train(1,:),out_SN.gamma_train(2,:),'ro');
% title(['training data fit: shallow NN, MSE=' num2str(out_SN.L_train)])
% subplot(2,3,1);plot(pi_train(1,:),pi_train(2,:),'.');hold on;plot(y_train(1,:),y_train(2,:),'ro')
% title(['training data fit: RFE, MSE=' num2str(RF_AUC_train(i_prop,i_ann))])
% subplot(2,3,5);plot(pi_valid(1,:),pi_valid(2,:),'.');hold on;plot(out_SN.gamma_test(1,:),out_SN.gamma_test(2,:),'ro');
% title(['test data fit: shallow NN, MSE=' num2str(out_SN.L_test)])
% subplot(2,3,4);plot(pi_valid(1,:),pi_valid(2,:),'.');hold on;plot(y_predict(1,:),y_predict(2,:),'ro')
% title(['test data fit: RFE, MSE=' num2str(RF_AUC(i_prop,i_ann))])
% subplot(2,3,3);plot(pi_train(1,:),pi_train(2,:),'.');hold on;plot(out_SPARTA.gamma_train(1,:),out_SPARTA.gamma_train(2,:),'ro');
% title(['training data fit: SPARTA, MSE=' num2str(out_SPARTA.L_train)])
% subplot(2,3,6);plot(pi_valid(1,:),pi_valid(2,:),'.');hold on;plot(out_SPARTA.gamma_test(1,:),out_SPARTA.gamma_test(2,:),'ro');
% title(['test data fit: SPARTA, MSE=' num2str(out_SPARTA.L_test)])

%     %% Run the DL LSTM algorithm
%     out_DL= DeepNN_Classify_ts(X_train,pi_train, X_valid, pi_valid,flag_AUC);
%     [~,i]=max(-out_DL.L_pred_valid);
%     LSTM_AUC(i_prop,i_ann)=max(-out_DL.L_pred_valid_ts(i))
%     t_LSTM(i_prop,i_ann)=out_DL.time(i);
%     if i_ann==1
%      Conf_matr_LSTM=(1/N_anneal)*out_DL.confusion_matr_valid_ts{i};
%     else
%      Conf_matr_LSTM= Conf_matr_LSTM+(1/N_anneal)*out_DL.confusion_matr_valid_ts{i};    
%     end 
% 
pause(0.2)
end

%% Visualizing the results
figure;subplot(2,1,1);EB(1:100,ml(:,(2+enso_lag):(1+enso_lag+100)),'-')
set(gca,'FontSize',24,'LineWidth',2,'XScale','log');
subplot(2,1,2);EB(1:100,ml(:,(1+enso_lag+101):size(ml,2)),'-')
set(gca,'FontSize',24,'LineWidth',2,'XScale','log');

    if flag_persistent_prediction==1;
E=[SPARTA_AUC' RF_AUC' SN_AUC' MP_AUC' PP_AUC' Lin_AUC' DL_AUC'];
E_t=[t_SPARTA' t_RF' t_SN' t_MP' t_PP' t_Lin' t_DL'];
model_label={'SPARTA','RFE','shallow NN','mean predictor','persist. prediictor', 'multilinear (l1 and l2)', 'DL with LSTM'}
[~,ii]=sort(mean(E),'ascend');
figure;subplot(2,1,1);h=boxplot(E(:,ii),'Notch','on','Labels',model_label(ii),'Whisker',1,'FullFactors','on')
set(h,{'linew'},{2})
set(gcf,'Position',[10 100 800  600]);
ylabel('MSE on test data');
title('Mean Squared Euclidean Prediiction Error (MSE) on test data')
%ylim([0.5 1])
set(gca,'FontSize',20,'LineWidth',2,'YScale','linear');
subplot(2,1,2);h=boxplot(E_t(:,ii),'Notch','on','Labels',model_label(ii),'Whisker',1,'FullFactors','on')
set(h,{'linew'},{2})
set(gcf,'Position',[10 100 1200  900]);
ylim([0.001 1e3])
ylabel('CPU time (sec.)')
title('Cost Comparison')
set(gca,'FontSize',20,'LineWidth',2,'YScale','log','YTick',[1e-3 1e-2 1e-1 1e0 1e1 1e2]);
    else
E=[SPARTA_AUC' RF_AUC' SN_AUC' MP_AUC' Lin_AUC' DL_AUC'];
E_t=[t_SPARTA' t_RF' t_SN' t_MP' t_Lin' t_DL'];
model_label={'SPARTA','RFE','shallow NN','mean predictor', 'multilinear (with l1 and l2 reg.)', 'DL with LSTM'}
[~,ii]=sort(mean(E),'ascend');
figure;subplot(2,1,1);h=boxplot(E(:,ii),'Notch','on','Labels',model_label(ii),'Whisker',1,'FullFactors','on')
set(h,{'linew'},{2})
set(gcf,'Position',[10 100 800  600]);
ylabel('MSE on test data');
title('Mean Squared Euclidean Prediiction Error (MSE) on test data')
%ylim([0.5 1])
set(gca,'FontSize',20,'LineWidth',2,'YScale','linear');
subplot(2,1,2);h=boxplot(E_t(:,ii),'Notch','on','Labels',model_label(ii),'Whisker',1,'FullFactors','on')
set(h,{'linew'},{2})
set(gcf,'Position',[10 100 1200  900]);
ylim([0.001 1e3])
ylabel('CPU time (sec.)')
title('Cost Comparison')
set(gca,'FontSize',20,'LineWidth',2,'YScale','log','YTick',[1e-3 1e-2 1e-1 1e0 1e1 1e2]);
    end        

    delta_E=max(bsxfun(@minus,E,min(E')'),1e-12);
[~,ii]=sort(median(delta_E),'ascend');
figure;h=boxplot(delta_E(:,ii),'Notch','on','Labels',model_label(ii),'Whisker',1,'FullFactors','on')
set(h,{'linew'},{2})
set(gcf,'Position',[10 100 800  600]);
%set(gca,'TickLabelInterpreter','latex')
ylabel('Deviation to the model with the min. MSE')
set(gca,'FontSize',20,'LineWidth',2,'XScale','linear','YScale','log');ylim([5e-13 1e-2])
ax = gca;


    W_mean=0*W_SPARTA{1};for n=1:numel(W_SPARTA);W_mean=W_mean+(1/numel(W_SPARTA))*W_SPARTA{n};end
for i=1:numel(W_SPARTA);W(i,:)=W_SPARTA{i};end
figure;EB(1:length(W_mean),W,'--');hold on;plot(W_mean,'k-','LineWidth',3);
mean_pattern=bsxfun(@times,sqrt(mean(W))',out_SPARTA.C);
[pred] = SPARTA_output_C(out_SPARTA);[mm,ii]=sort(pred(1,:));
[XX,YY]=meshgrid((mm*max_pi(1)+min_pi(1)),1:size(out_SPARTA.W,1));
figure;surf(XX,YY,mean_pattern(:,ii));

Lambda=SPARTA_output_Lambda(out_SPARTA,1);
figure;subplot(1,2,1);semilogx((Lambda(:,(2+enso_lag):(2+enso_lag+100))'),':.','LineWidth',2,'MarkerSize',24)
set(gca,'FontSize',20,'LineWidth',2,'YScale','linear')
legend;
axis tight;ylim([-0.4 0.4])
subplot(1,2,2);semilogx((Lambda(:,(2+enso_lag+101):size(Lambda,2))'),':.','LineWidth',2,'MarkerSize',24)
set(gca,'FontSize',20,'LineWidth',2,'YScale','linear');
axis tight;ylim([-0.4 0.4])
legend;
set(gcf,'Position',[10 100 800  600]);

%figure;plot(max_pi(1)*pi_valid+min_pi(1),'-.','MarkerSize',10,'LineWidth',3);hold on;plot(max_pi(1)*out_SPARTA.gamma_test+min_pi(1),'k--o','MarkerSize',10,'LineWidth',2);plot(max_pi(1)*out_DL.gamma_valid+min_pi(1),'r:sq','MarkerSize',10,'LineWidth',2);
t=(2006.45-(length(out_SPARTA.gamma_valid))/12)+ [1:length(out_SPARTA.gamma_valid)]./12;
figure;plot(t,max_pi(1)*pi_valid+min_pi(1),'-.','MarkerSize',10,'LineWidth',3);hold on;plot(t,max_pi(1)*out_SPARTA.gamma_test+min_pi(1),'k--o','MarkerSize',10,'LineWidth',2);plot(t,max_pi(1)*out_DL.gamma_valid +min_pi(1),'r:sq','MarkerSize',10,'LineWidth',2);
legend({'ENSO','SPARTA','Deep Learning CNN'});
set(gca,'FontSize',20,'LineWidth',2);
ylabel('ENSO');xlabel('Year');
set(gcf,'Position',[10 100 800  600]);

Lambda=SPARTA_output_Lambda(out_SPARTA,1);
figure;subplot(2,1,1);semilogx((Lambda(:,(2+enso_lag):(2+enso_lag+100))'),':.','LineWidth',2,'MarkerSize',24)
set(gca,'FontSize',24,'LineWidth',2);
subplot(2,1,2);semilogx((Lambda(:,(2+enso_lag+101):size(Lambda,2))'),':.','LineWidth',2,'MarkerSize',24)
set(gca,'FontSize',24,'LineWidth',2);

figure;subplot(2,1,1);semilogx((W_mean((1+enso_lag):(1+enso_lag+100))'),':.','LineWidth',2,'MarkerSize',24)
set(gca,'FontSize',24,'LineWidth',2);
subplot(2,1,2);semilogx((W_mean((1+enso_lag+101):length(W_mean))'),':.','LineWidth',2,'MarkerSize',24)
set(gca,'FontSize',24,'LineWidth',2);


figure;plot((Lambda'),':.','LineWidth',2)
set(gca,'FontSize',20,'LineWidth',2,'YScale','linear');

figure;EB(x,SPARTA_Bin,'-');hold on;EB(x,SN_Bin,'--');EB(x,DL_Bin,':');EB(x,Lin_Bin,'-.');EB(x,RF_Bin,'-');EB(x,PP_Bin,'-.');EB(x,MP_Bin,':');
set(gca,'FontSize',20,'LineWidth',2,'YScale','linear');
xlim([0 1])
legend('SPARTA','shallow NN', 'DL with LSTM', 'multilinear (with l1 and l2 reg.)','RFE','persistent predictor','mean predictor')
set(gca,'FontSize',20,'LineWidth',2,'YScale','log');

load Output/ii
[~,ii]=sort(mean(E),'ascend');figure;EB(1:7,E(:,ii),':');

