function [ind_s,ind_ns]= UniformSubsample(X,N);
minX=min(X);maxX=max(X);
T=length(X);
ind_s=[];
ind_ns=1:T;
for n=1:N
    r=minX+(maxX-minX)*rand(1);
    [~,i]=min(abs(r-X(ind_ns)));
    ind_s=[ind_s ind_ns(i)];
    ind_ns=setdiff(ind_ns,ind_ns(i));
end