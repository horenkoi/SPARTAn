function [] = VisualizeBox(X,gamma)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[K,T]=size(gamma);
col='rgbmcky';
mark='x.o<>h*';
hold on;
for k=1:K
    ind_mark=ceil(k/numel(mark));
    ind_col=k-(ind_mark-1)*numel(mark);
    ii=find(gamma(k,:)==1);
    plot(X(1,ii),X(2,ii),[mark(ind_mark) col(ind_col)]);  
end
end