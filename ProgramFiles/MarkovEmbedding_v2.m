function [XX,YY] = MarkovEmbedding_v2(X,emb_dim)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[N,T]=size(X);
XX=zeros(N*emb_dim,T-emb_dim-1);YY=zeros(N,T-emb_dim-1);
for t=emb_dim:(T-1)
    for i=1:emb_dim
       XX(N*(i-1)+1:N*i,t-emb_dim+1)=X(:,t-i+1); 
    end
    YY(:,t-emb_dim+1)=X(:,t+1); 
end    

end

