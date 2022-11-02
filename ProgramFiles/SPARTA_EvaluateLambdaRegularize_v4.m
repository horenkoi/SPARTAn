%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of Lambda in SPARTA
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 


function [Lambda]=SPARTA_EvaluateLambdaRegularize_v4(Lambda0,X,W,pi,idx,COV,m,K,N,T,eps_CL,eps_L2);

Lambda=zeros(m,N+1,K);
%ii=find([1 W]<1e-2);pc=zeros(1,N+1);pc(ii)=1;PC=sparse(pc'*pc);

eps=eps_L2*T/(N*eps_CL);
for k=1:K
      mu=sum(X(:,idx{k}),2);
      COV_new=([1 W]'*[1 W]).*[length(idx{k}) mu';mu COV{k}];
      XW=[ones(1,length(idx{k})); X(:,idx{k})];
      for ind=1:m
      [Lambda(ind,:,k),~,~,iter]=pcg((COV_new+eps*eye(N+1)),(bsxfun(@times,[1 W]',(XW*pi(ind,idx{k})'))),[],[],[],[],Lambda0(ind,:,k)');
      end
end
%beta=((Xw*Xw'+eps_l2*T*eye(m))\(Xw*pi_train'))';
