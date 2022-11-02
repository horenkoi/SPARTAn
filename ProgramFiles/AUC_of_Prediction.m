%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computation of the accuracy in the prediction
%%
%% Comparison of eSPA+ (advanced entropic Scalable Probabilistic Approximation Algoriithm)
%% to the Deep Learning with Long-Short Term Memory algorithm (DL LSTM) and
%% the shallow neuronal networks (shallow NN) for diifferent numbers of neurons in the hidden layer
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% 
% Input - gamma: "K-times-T" matrix of box probabilities
%         P: Lambda
%         pi_valid: matrix containing the lables (validation set)
%         flag_nn: set by default equal to zero
%         flag_AUC: if 1, the accuracy is computed using the Area Under the ROC Curve


function [AUC] = AUC_of_Prediction(gamma,P,pi_valid,flag_nn,flag_AUC)
    if flag_nn==0
        xx = P * gamma;
    else
        xx = P(gamma);
    end

    if flag_AUC == 1
        % Compute the area under the ROC curve
        [~,~,~,AUC] = perfcurve(pi_valid(1,:)',xx(1,:)',1);
        AUC = sum(AUC);
        if AUC < 0.5
            AUC= 1 - AUC;
        end
        AUC = -AUC * size(pi_valid,1) * size(pi_valid,2);
    else
        % Computation of the Accuuracy
        AUC = 0;
        for t = 1:size(xx,2)
            AUC = AUC + MyAccuracy(pi_valid(:,t),xx(:,t));
        end
    end

end

