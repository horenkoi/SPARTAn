%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the value of the functional L in SPARTA
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
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


