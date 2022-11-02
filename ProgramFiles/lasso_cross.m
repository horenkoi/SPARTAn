function out= lasso_cross(X_train,pi_train, X_valid, pi_valid,reg_param,reg_ridge);
%options=optimset('GradObj','on','Algorithm','sqp','MaxIter',20,'Display','off','TolFun',1e-13,'TolCon',1e-13,'TolX',1e-13,...
%    'TolConSQP',1e-13,'TolGradCon',1e-13,'TolPCG',1e-13,'MaxFunEval',20000,'UseParallel',false);
%C_orig=diag(1./min(sqrt(W),1e-5))*C;
tresh=0.05;
tic;
X_train=[ones(1,size(X_train,2));X_train];X_valid=[ones(1,size(pi_valid,2));X_valid];
[m,T]=size(X_train);
for jj=1:size(pi_train,1)
for ind_ridge=1:length(reg_ridge)
    for lambda_ind=1:length(reg_param)
        [B,FitInfo] = lasso(X_train(2:m,:)',pi_train(jj,:)','Intercept',true,'Lambda',reg_param(lambda_ind)+2*reg_ridge(ind_ridge),'Alpha',reg_param(lambda_ind)./(reg_param(lambda_ind)+2*reg_ridge(ind_ridge)),'Standardize',false);
        out.beta(:,ind_ridge,lambda_ind)=[FitInfo.Intercept; B]';
        %tresh=par_tresh*max(abs(out.beta(:,ind_ridge,lambda_ind)));
        out.N_param(ind_ridge,lambda_ind)=length(find(abs(squeeze(out.beta(:,ind_ridge,lambda_ind)))>tresh))+2;
        y_valid{ind_ridge,lambda_ind}=out.beta(:,ind_ridge,lambda_ind)'*X_valid;
        out.error_pred(ind_ridge,lambda_ind)=(1/size(X_valid,2))*sum((pi_valid(jj,:)-y_valid{ind_ridge,lambda_ind}).^2);
        out.error_pred_train(ind_ridge,lambda_ind)=(1/size(X_train,2))*sum((pi_train(jj,:)-out.beta(:,ind_ridge,lambda_ind)'*X_train).^2);
    end
end
    minMatrix = min(out.error_pred_train(:));
    [row,col] = find(out.error_pred_train==minMatrix);
    out.min_par_err(jj)=out.error_pred(row(1),col(1));
    out.ridge(jj)=reg_ridge(row(1));
    out.lasso(jj)=reg_param(col(1));
    out.y_valid(jj,:)=y_valid{row(1),col(1)};

out.time(jj)=toc/(length(reg_param)*length(reg_ridge));
%W=(exp(-arg+max_arg)./sum(exp(-arg+max_arg)))';
end
out.min_par_err=sum(out.min_par_err);
out.time=sum(out.time);
