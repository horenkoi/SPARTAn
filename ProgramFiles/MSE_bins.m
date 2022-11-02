function [y,x] = MSE_bins(X,Y,bins)

x=bins(1:(length(bins)-1))+0.5*diff(bins);
for i=1:(length(bins)-1)
    ii=find(and(bins(i)<=X,X<bins(i+1)));
    y(i)=mean(Y(ii));  
end
end