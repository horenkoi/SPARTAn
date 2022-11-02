%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Analytical evaluaton of the gamma-step (p. 1567 eSPA paper)
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
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


function [gamma] = SPACL_EvaluateGamma(X,pi,C,Lambda,T,K,m,d,reg_param);
	
	% Computation of the gamma step (see p. 1567 eSPA paper) 	     % Why (1/d)?
	[~,idx] = min(-(1/m) * reg_param * log(max(Lambda',1e-12)) * pi + (1/d) * sqDistance(X, C)');

	% Initialization of the gamma vector
	gamma = zeros(K,T);

	% Checking if the condition of the analytical solution is met
	for k = 1:K
	    gamma(k,find(idx==k)) = 1;
	end 

end

