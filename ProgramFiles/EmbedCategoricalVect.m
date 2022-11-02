%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Embedding of multiiple categorical variables into one categorical vector of larger dimension
%%  Single categorical variable series are rows of input Y
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X,St_Y,P]= EmbedCategoricalVect(Y)

[N_tors,T]=size(Y);
X(1)=1; St_Y=Y(:,1);
for t1=2:T
    flag=0;
    for t2=1:size(St_Y,2)
        if St_Y(:,t2)==Y(:,t1)
           X(t1)=t2;
           flag=1;
        end
    end
    if flag==0
        X(t1)=size(St_Y,2)+1;
        St_Y=[St_Y Y(:,t1)];
    end
end
nn=max(X)-min(X)+1;mm=min(X)-1;
if nargout>2
P=zeros(nn,nn);
for t1=2:T
    P(X(t1-1)-mm,X(t1)-mm)=P(X(t1-1)-mm,X(t1)-mm)+1;
end

for i=1:nn
    P(i,:)=P(i,:)/sum(P(i,:));
end
P=P';
end

% for j=1:size(St_Y,2)
%     mm(j)=sum(X==j);
% end
% 
% [MM,II]=sort(mm,'descend');
% Stf=St_Y(:,II);
% Xf=zeros(N,T);
% for j=1:size(Stf,2)
%     if j<N
%     Xf(j,find(II(j)==X))=1;
%     else
%     Xf(N,find(II(j)==X))=1;    
%     end
%end