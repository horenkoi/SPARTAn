%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the value of the functional L in SPARTA
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 



function [L,y_pred] = SPARTA_pred_error(X,pi,Lambda,gamma,T,m, W,K);

regr_dist=0;
for k=1:K
    idx{k}=find(gamma(k,:)==1);
    if ~isempty(idx{k})
        for j=1:m
            y_pred(j,idx{k})=Lambda(j,:,k)*[ones(1,length(idx{k})); bsxfun(@times,W',X(:,idx{k}))];
            regr_dist=regr_dist+sum((pi(j,idx{k})-y_pred(j,idx{k})).^2);
        end
    end
end

	% Updating the main functional 
	L = 1/T * regr_dist;

end


