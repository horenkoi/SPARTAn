function [pred] = SPARTA_output_Lambda(out,i)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    %XW=bsxfun(@times,out.W',out.C'); 
    m=size(out.Lambda,1);[d,K]=size(out.C);
    W=[1 out.W'];
    pred=zeros(K,d+1);
    for k=1:K
        for j=1:(d+1)
            pred(k,j)=out.Lambda(i,j,k)*W(j);             
        end
    end
end