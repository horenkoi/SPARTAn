%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Analytical computation of the W-step (p. 1567 eSPA paper)
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input - X: matrix containing the features
%         gamma: matrix of box probabilities
% 		  C: matrix of box coordinates 
%		  d: number of features in the matrix X
% 		  T: size of the data statistic 
%         W:  vector of feature probabilities 
%         eps_C: value of the regularizzation parameter epsilon_C 

function [W] = eSPA_matrix_W_EvaluateWRegularize(X,gamma,C,d,T,W,eps_C)
    xx=X-C*gamma;
    xx=xx*xx';xx=xx.*(2*ones(d)-eye(d));
    k=1;
    b=zeros(0.5*d*(d+1),1);
    for i=1:d
        for j=i:d
            b(k)=xx(i,j);  
            k=k+1;
        end
    end
	%b = sum((X-C*gamma).^2,2);
    bmin=min(b);b=b-bmin;
	z = exp(-b./(T*eps_C));

	
	w = (z./sum(z))';
    W=zeros(d);
    k=1;
    for i=1:d
        W(i,i)=w(k);
        k=k+1;
        for j=(i+1):d
            W(i,j)=w(k);
            W(j,i)=W(i,j);
            k=k+1;
        end
    end

end


