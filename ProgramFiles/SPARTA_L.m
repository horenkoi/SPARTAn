%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the value of the functional L in SPARTA
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 



function [L] = SPARTA_L(X,pi,C,Lambda,gamma,T,d,m, reg_param, eps_C,W,K,eps_L2);
     
    regr_dist=0;
    disc_dist=0;
    for k=1:K
      idx=(gamma(k,:)==1);
      disc_dist=disc_dist+norm(bsxfun(@times,sqrt(W'),bsxfun(@minus,X(:,idx),C(:,k))),'fro')^2;
      for j=1:m
            regr_dist=regr_dist+sum((pi(j,idx)-Lambda(j,1,k)-Lambda(j,2:(d+1),k)*bsxfun(@times,W',X(:,idx))).^2);             
        end
    end

	% Updating the main functional 
	L = disc_dist * (1/T) + eps_C * sum(W.*log(max(W,1e-12))) + ...
        reg_param/(T*m) * regr_dist+eps_L2/(d*m)*sum(sum(sum(Lambda.^2)));

end


