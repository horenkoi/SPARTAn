%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of the W-step in eSPA
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input - X: matrix containing the features
%         gamma: matrix of box probabilities
% 		  C: matrix of box coordinates 
%		  d: number of features in the matrix X
% 		  T: size of the data statistic 
%         W:  vector of feature probabilities 
%         eps_C: value of the regularizzation parameter epsilon_C 

function [W] = SPACL_dim_entropy_EvaluateWRegularize_v3(X,gamma,C,d,T,W,eps_C)

	b = sum((X-C*gamma).^2,2);
	z = exp(-b./(T*eps_C));

	% new improved version
	W = (z./sum(z))';

end


