%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the value of the functional L according to eq. 2.4 p. 1571 eSPA paper
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 

% Input - X: matrix containing the features
%		  pi: matrix containing the lables 
% 		  C: matrix of box coordinates 
% 		  Lambda: matrix of conditional probabilites 
%         gamma: matrix of box probabilities
% 		  T: size of the data statistic 
%		  m: number of labels
%		  d: number of features in the matrix X
%		  reg_param: matrix with the combination of the regularization parameters
%         eps_C: value of the regularizzation parameter epsilon_C
%         W: vector of feature probabilities
%		  K: number of discretization boxes
%         eps_Creg: set by default equal to 1e-10



function [L] = EOS_SPA_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m, reg_param, eps_C,W,K,eps_Creg,alpha);
	
	% Updating the main functional 
	L = norm((X - C*gamma),'fro')^2 * (1/(T*d)) + (eps_C/d) * sum(W.*log(max(W,1e-12))) - reg_param/(T*m) * sum(sum(pi.*(log(max(Lambda*gamma,1e-12)))));


	Ls = 0;
	for i = 1:d
	    for k1 = 1:K
	        for k2 = 1:K
	            Ls = Ls + (C(i,k1) - C(i,k2))^2;
	        end
	    end
	end
	Ls = 1/(T*d*K*(K-1)) * Ls;
	L = L + eps_Creg*Ls;
end


