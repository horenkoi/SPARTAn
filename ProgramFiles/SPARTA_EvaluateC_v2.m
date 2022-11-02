%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Analytical evaluaton of the C-step in SPARTA
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




function [C] = SPARTA_EvaluateC_v2(X,idx,K,d);

% Initialization of an empty vector
C = zeros(d,K);

% Sum over the rows (i.e., over t) of the gamma matrix
for k=1:K
    C(:,k)=(mean(X(:,idx{k})'))';
end

end
