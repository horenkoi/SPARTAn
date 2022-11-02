%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the value of the functional L in SPARTA
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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


