function [COV,idx] = COV_Update(X,ind1,ind2,COV)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin<3
    for k=1:size(ind1,1)
        idx{k}=find(ind1(k,:)==1);
        COV{k}=X(:,idx{k})*X(:,idx{k})';
    end
else
    idx=[];
    for k=1:numel(ind2)
        idx_out=setdiff(ind2{k},ind1{k});
        idx_in=setdiff(ind1{k},ind2{k});
        COV{k}=COV{k}-X(:,idx_out)*X(:,idx_out)'+X(:,idx_in)*X(:,idx_in)';
    end
end