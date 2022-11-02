%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Vuisualisation of the eSPA Voronoi diagramm in 2D
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function []=VisualizeVoronoi_v2(C,P)

C=C(1:2,:);W=[1 1]';
N_grid=100;
[XX,YY]=meshgrid(min(C(1,:)):(max(C(1,:))-min(C(1,:)))/N_grid:max(C(1,:)),min(C(2,:)):(max(C(2,:))-min(C(2,:)))/N_grid:max(C(2,:)));
xx=reshape(XX,1,numel(XX));yy=reshape(YY,1,numel(YY));

[~,idx] = min(sqrt(sqDistance(bsxfun(@times,sqrt(W),[xx; yy]), bsxfun(@times,sqrt(W),C))'));
zz=0*xx;
for t=1:length(idx)
zz(t)=P(idx(t));
end
hold on;
[~, contourObj]=contourf(XX,YY,reshape(zz,size(XX,1),size(XX,2)),P);
h = voronoi(C(1,:)',C(2,:)');hold on;
h(1).MarkerSize=15;h(1).Marker='o';h(1).MarkerFaceColor='k';
set(h(2:end),{'linew'},{2});set(h(2:end),{'color'},{'r'});set(h(2:end),{'linestyle'},{':'})
set(gcf,'Position',[10 100 800  600]);
set(gca,'FontSize',24,'LineWidth',2);
%axis off
colormap jet
caxis([0 1])
end
function updateTransparency(contourObj)
contourFillObjs = contourObj.FacePrims;
for i = 1:length(contourFillObjs)
    % Have to set this. The default is 'truecolor' which ignores alpha.
    contourFillObjs(i).ColorType = 'truecoloralpha';
    % The 4th element is the 'alpha' value. First 3 are RGB. Note, the
    % values expected are in range 0-255.
    contourFillObjs(i).ColorData(4) = 30;
end
end
