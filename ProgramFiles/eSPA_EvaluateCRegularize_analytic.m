%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Analytical evaluaton of the S-step (p. 1567 eSPA paper)
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Input - X: "d times T" matrix containing the features
%		  gamma:  "K-times-T" matrix of box probabilities
% 		  K: number of geometric boxes
% 		  d: number of features in the matrix X
% 		  T: size of the data statistic 



function [C] = eSPA_EvaluateCRegularize_analytic(X,gamma,K,d,T,W);

C=(W*X*gamma')'\(gamma*gamma');
C=(C\W)';
% %	Initialization of an empty vector 
% 	C = zeros(d,K); 
% 
% 	% Sum over the rows (i.e., over t) of the gamma matrix
% 	N = sum(gamma',1);
% 
% 	% iteration for each geometric box K
% 	for k = 1:K
% 
% 		% Perform the sum for the numerator of the analytical solution
% 	    for t = 1:T
% 	        C(:,k) = C(:,k) + gamma(k,t)*X(:,t);
% 	    end
% 
% 	    % Perform the analytical solution for each geometric box
% 	    if N(k) > 0
% 	    	C(:,k) = C(:,k)./N(k);
% 	    else     % Is this a check for a division by zero? Is it even possible?
% 	    	C(:,k) = (1e-10)*randn(d,1);    
% 	    end
% 
% 	end
% 
end
