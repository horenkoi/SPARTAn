%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of Lambda in SPARTA
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 


function [Lambda]=SPARTA_EvaluateLambdaRegularize(X,W,pi,gamma,m,K,N,T,eps_CL,eps_L2);

Lambda=zeros(m,N+1,K);
eps=eps_L2*T/(N*eps_CL);
for k=1:K
      idx{k}=find(gamma(k,:)==1);
      XW=[ones(1,length(idx{k})); bsxfun(@times,W',X(:,idx{k}))];
      Lambda(:,:,k)=((XW*XW'+eps*eye(N+1))\(XW*pi(:,idx{k})'))';
end
%beta=((Xw*Xw'+eps_l2*T*eye(m))\(Xw*pi_train'))';