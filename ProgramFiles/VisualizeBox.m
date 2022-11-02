%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%

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