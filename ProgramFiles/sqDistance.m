%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Optimially-vectorized computation of the squared Euclidean distances for matrix data
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 

% Input - X and Y: matrices with coherent dimensions

function D = sqDistance(X, Y)
D = bsxfun(@plus, dot(X,X,1)' , dot(Y,Y,1)) - 2*(X'*Y);
