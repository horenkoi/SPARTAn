%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%

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