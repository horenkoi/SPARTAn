%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Evaluation of the first part of the functional L (used for the computation of the accuracy)
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 
% Input - X: matrix containing the features
%		  C: matrix of box coordinates
%		  T: size of the data statistic
%		  K: number of discretization boxes

function [gamma] = SPACL_EvaluateGamma_valid(X,C,T,K)

	[~,idx] = min(sqDistance(X, C)');
	gamma = zeros(K,T);
	for k = 1:K
	    gamma(k,find(idx==k)) = 1;
	end
	
end

