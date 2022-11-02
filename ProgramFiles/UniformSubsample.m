%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%

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