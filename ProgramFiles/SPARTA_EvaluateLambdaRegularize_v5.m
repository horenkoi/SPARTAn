%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of Lambda in SPARTA
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%


function [Lambda]=SPARTA_EvaluateLambdaRegularize_v5(Lambda0,X,W,pi,gamma,m,K,N,T,eps_CL,eps_L2);
warning('off')
global XW WW N TW eps
WW=[1 W]';
Lambda=zeros(m,N+1,K);
%ii=find([1 W]<1e-2);pc=zeros(1,N+1);pc(ii)=1;PC=sparse(pc'*pc);
myA=@afun;

eps=eps_L2*T/(N*eps_CL);
for k=1:K
    idx=(gamma(k,:)==1);
    %mu=sum(X(:,idx{k}),2);
    %COV_new=([1 W]'*[1 W]).*[length(idx{k}) mu';mu COV{k}];
    TW=sum(gamma(k,:));
    XW=[ones(1,TW); X(:,idx)];
    %COV_new=([1 W]'*[1 W]).*(XW*XW');
    for ind=1:m
        %[Lambda(ind,:,k),~,~,iter]=pcg((COV_new+eps*eye(N+1)),(bsxfun(@times,[1 W]',(XW*pi(ind,idx{k})'))),[],[],[],[],Lambda0(ind,:,k)');
        [Lambda(ind,:,k),~,~,iter]=pcg(myA,bsxfun(@times,[1 W]',(XW*pi(ind,idx)')),[],[],[],[],Lambda0(ind,:,k)');
    end
end
%beta=((Xw*Xw'+eps_l2*T*eye(m))\(Xw*pi_train'))';
end

function y=afun(x)
global XW WW N TW eps
y=0*x;
xw= WW.*x;
for t=1:TW
    y=y+XW(:,t)*(XW(:,t)'*xw);
end
y=WW.*y+eps*x;
end
