%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of Lambda in SPARTA
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 


function [Lambda]=SPARTA_EvaluateLambdaRegularize_v3(Lambda0,X,W,pi,idx,COV,m,K,N,T,eps_CL,eps_L2);

options = optimoptions(@quadprog,...
    'Algorithm','active-set',...
    'MaxIterations',5,'Display','off');

Lambda=zeros(m,N+1,K);
eps=eps_L2*T/(N*eps_CL);
for k=1:K
      mu=sum(X(:,idx{k}),2);
      COV_new=([1 W]'*[1 W]).*[length(idx{k}) mu';mu COV{k}];
      XW=[ones(1,length(idx{k})); X(:,idx{k})];
      for ind=1:m
          H=2*(0.5*COV_new + 0.5*COV_new'+eps*eye(N+1));
          f=-([1 W]'.*(XW*pi(ind,idx{k})'))';
          fun0=0.5*Lambda0(ind,:,k)*H*Lambda0(ind,:,k)'+f*Lambda0(ind,:,k)';
          [Lambda(ind,:,k),fval,exitflag,output] = quadprog(H,f,[],[],[],[],[],[],Lambda0(ind,:,k),options); 
          if fun0<fval
             Lambda(ind,:,k)= Lambda0(ind,:,k);
          end
      %Lambda(:,:,k)=((COV_new+eps*eye(N+1))\(bsxfun(@times,[1 W]',(XW*pi(:,idx{k})'))))';
      end
end
%beta=((Xw*Xw'+eps_l2*T*eye(m))\(Xw*pi_train'))';
