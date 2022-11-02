
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Analytical evaluaton of the C-step in SPARTA
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [C] = SPARTA_EvaluateC_v2(X,idx,K,d);

% Initialization of an empty vector
C = zeros(d,K);

% Sum over the rows (i.e., over t) of the gamma matrix
for k=1:K
    C(:,k)=(mean(X(:,idx{k})'))';
end

end
