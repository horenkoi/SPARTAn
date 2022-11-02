
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Analytical evaluaton of the gamma-step (p. 1567 eSPA paper)
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 
% Input - X: matrix containing the features
%		  pi: matrix containing the lables 
% 		  C: matrix of box coordinates 
% 		  Lambda: matrix of conditional probabilites 
% 		  T: size of the data statistic 
%		  K: number of discretization boxes
%		  m: number of labels
%		  d: number of features in the matrix X
%		  reg_param: matrix with the combination of the regularization parameters


function [gamma] = SPARTA_EvaluateGamma(X,pi,C,Lambda,T,K,m,d,reg_param,W);
	
	% Computation of the gamma step (see p. 1567 eSPA paper) 	     % Why (1/d)?
	%[~,idx] = min(-(1/m) * reg_param * log(max(Lambda',1e-12)) * pi + (1/d) * sqDistance(X, C)');

    XW=bsxfun(@times,W',X); 
    regr_dist=zeros(K,T);
    for k=1:K
        for j=1:m
            regr_dist(k,:)=regr_dist(k,:)+(pi(j,:)-Lambda(j,1,k)-Lambda(j,2:(d+1),k)*XW).^2;             
        end
    end
    if K>1
        [~,idx] = min((1/m) * reg_param * regr_dist + sqDistance(bsxfun(@times,sqrt(W'),X), bsxfun(@times,sqrt(W'),C))');
    else
        idx=ones(1,T);
    end
	% Initialization of the gamma vector
	gamma = sparse(K,T);

	for k = 1:K
	    gamma(k,find(idx==k)) = 1;
	end 

end

