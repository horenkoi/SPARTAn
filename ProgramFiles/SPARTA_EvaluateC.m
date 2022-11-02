
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Analytical evaluaton of the C-step in SPARTA
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [C] = SPARTA_EvaluateC(X,gamma,K,d,T);

% Initialization of an empty vector
C = zeros(d,K);

% Sum over the rows (i.e., over t) of the gamma matrix
for k=1:K
    idx=(gamma(k,:)==1);
    C(:,k)=(mean(X(:,idx),2));
end

end
