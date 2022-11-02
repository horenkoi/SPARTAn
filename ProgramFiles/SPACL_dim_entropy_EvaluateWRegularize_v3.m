%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of the W-step in eSPA
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
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


