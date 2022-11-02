function [pred] = SPARTA_output_C(out)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    XW=bsxfun(@times,out.W',out.C'); 
    m=size(out.Lambda,1);[d,K]=size(out.C);
    pred=zeros(m,K);
    for k=1:K
        for j=1:m
            pred(j,k)=out.Lambda(j,1,k)+sum(out.Lambda(j,2:(d+1),k).*XW(k,:));             
        end
    end
end