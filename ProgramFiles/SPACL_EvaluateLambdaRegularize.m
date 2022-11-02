%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of Lambda using the formula Pi*gamma in eSPA
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 

% Input - pi: matrix containing the lables
%		  gamma: matrix of box probabilities
%		  m: number of labels
%		  K: number of discretization boxes

function [Lambda,p]=SPACL_EvaluateLambdaRegularize(pi,gamma,m,K);

Lambda0 = pi * gamma';
Lambda=0*Lambda0;
p=zeros(1,K);
% Compute the analytical solution for each geometrix box
for k = 1:K
    ss = sum(Lambda0(:,k));
    if ss > 0
        Lambda(:,k) = Lambda0(:,k) ./ sum(Lambda0(:,k));
    else
        Lambda(:,k)=1/m;
    end
end
if nargout>1
    [~,ii1]=max(pi);[~,ii2]=max(gamma);
    for k = 1:K
        [tbl,chi2,p(k)] = crosstab(ii1,(ii2==k));
        if sum(size(tbl))==4
        [h,p(k),stats] = fishertest(tbl);
        end
    end
end
end


