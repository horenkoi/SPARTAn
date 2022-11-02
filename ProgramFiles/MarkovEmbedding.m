function [Y] = MarkovEmbedding(X,emb_dim)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[N,T]=size(X);
Y=zeros(N*emb_dim,T-emb_dim+1);
for t=emb_dim:T
    for i=1:emb_dim
       Y(N*(i-1)+1:N*i,t-emb_dim+1)=X(:,t-i+1); 
    end
end    

end

