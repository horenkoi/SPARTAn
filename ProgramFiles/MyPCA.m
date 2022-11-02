%% Computation of the PCA dimension reduction for the 
%% data matrix X
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0

function [X_proj,V,mu,D]=MyPCA(X,K);

[V,D]=eigs(cov(X'),K);
mu=mean(X,2);
[N,T]=size(X);
X_proj=V'*(X-repmat(mu,1,T));
X_proj=X_proj';