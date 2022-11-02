%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% eSPA main algorithm with a triple split (into training, validation and testing data)
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Input - X: "d times T" matrix containing the features
%        Pi: "m times T" matrix containing the lables
%        K: number of discretization boxes
%        N_anneal: number of simulated annealing steps
%        out_init: initialization of gamma and Lambda, not necessary
%        reg_param: matrix with the combination of the regularization parameters
%        X_valid: validation set of the feature matrix X
%        pi_valid: validation set of the labels matrix Pi
%        N_neurons: Probably not used (Check)
%        flag_AUC: if 1, the accuracy is computed using the Area Under the ROC Curve
%        flag_parallel: if 1, the combination of regularizatio parameters are computed in parallel


function [out_opd] = SPARTA_v2(X,Pi,K,N_anneal,out_init,reg_param,X_valid,pi_valid,flag_parallel)
    
    % Creating variables containing the dimensions of X and Pi
    %[~,Viterbi_train]=max(Pi);
    [d,T]=size(X); m=size(Pi,1);
    for dd=1:d
        for mm=1:m
            R = corrcoef(X(dd,:),Pi(mm,:));
            if ~isnan(abs(R(1,2)))
                cc(mm)=abs(R(1,2));
            else
                cc(mm)=0; 
            end
        end
        w_init(dd)=max(cc);
    end
    w_init=w_init./sum(w_init);
    %[~,ii]=max(w_init);w_init=zeros(1,d);w_init(ii)=1;
    % Initialization of an iteration variable
    kkk = 1;
    for n = 1:N_anneal

        % Initialization for the stochastic matrix gamma (K-times-T)
        %gamma = rand(K,T);  

        % Make the matrix stochastic (sum of the columns equal to one)
        %for t = 1:T
        %    gamma(:,t) = gamma(:,t) ./ sum(gamma(:,t));
        %end

        % Initialization of gamma as input to the function
        %if and(~isempty(out_init),n==1)
        %    gamma = out_init.gamma;
        %end


        % Initialization of the matrix of conditional probabilities Lambda (m-times-K)
        % Lambda is a stochastich matrix so, every column must sum up to one
        Lambda = rand(m,d+1,K);
%         for k = 1:K
%              Lambda(:,k) = Lambda(:,k) ./ sum(Lambda(:,k));
%         end

        % Initialization of Lambda as input to the function
        if and(~isempty(out_init),n==1)
            Lambda = out_init.P;
        end

        % Initialization of C
        ind=randperm(T);
        C = X(:,ind(1:K));      


        % Different initialization for W as a stochastic 1-times-d vector
        if n>ceil(N_anneal/3)
            W = rand(1,d); W = W/sum(W);
        else
            W=w_init;
        end

        % For each combination of regularise parameters and for each annealing step,
        % create a structure with all the relevant data, the structure will have N_annealing * D cells
        for e = 1:size(reg_param,2)

            in{kkk}.X = X;
            %in{kkk}.gamma = gamma;
            in{kkk}.W = W;
            in{kkk}.Pi = Pi;
            in{kkk}.X_valid = X_valid;
            in{kkk}.pi_valid = pi_valid;
            in{kkk}.T = T;
            in{kkk}.K = K;
            in{kkk}.d = d;
            in{kkk}.Lambda = Lambda;
            in{kkk}.C = C;   % can be avoided
            in{kkk}.reg_param = reg_param(:,e);
            in{kkk}.e = e;
            in{kkk}.n = n;
            in{kkk}.N_anneal = N_anneal;
            kkk = kkk + 1;

        end
    end


    %_________________________________________________________________________________________________________


    % Start the time for the computation
    totalTimeW = 0;

    % Main body of the function, it compute the four steps (S, gamma, Lambda and W)
    % for ecah combiation of regularise parameters, in parallel or not depending on the flag
    if flag_parallel
        % Parallel version
        parfor kkk = 1:numel(in)
            [out{kkk}] = SPARTA_Replica_ts_v2(in{kkk});
            totalTimeW = totalTimeW + out{kkk}.timeW;
        end
    else
        % Serial version
        for kkk = 1:numel(in)
            [out{kkk}] = SPARTA_Replica_ts_v2(in{kkk});
            totalTimeW = totalTimeW + out{kkk}.timeW;
        end
    end
    %______________________________________________________________________________________

    for kkk = 1:numel(in)

        % Counts the number of anneal
        n = in{kkk}.n;     
        % Counts the pair of regularization parameters
        e = in{kkk}.e;     
        % Value of the functional in the training set
        % Value of the functional in the prediction ?
        L_pred(n,e) = out{kkk}.L_valid;
        %L_pred_ts(n,e) = out{kkk}.L_pred_Markov_ts;
        % Matrix containing 
        KKK(n,e) = kkk;
        % Matrix containing the computational time of W for
        % each combination of regularization paraeters and annealing step
        time(n,e) = out{kkk}.time;
    end

    % For each combination of regularization paraeters do:
    for e = 1:size(L_pred,2)

        % Select the minimum value in the annealing step
        [lll(e),ii] = min(L_pred(:,e));
        %lll_ts(e)=L_pred_ts(ii,e);

        % All the values below are selected based on the aforementioned
        % minimum value of the prediction in the annealing steps

        % Select the value of the fuctional L
        L_fin(e) = out{KKK(ii,e)}.L_train;
        % Select the value of the functional in the training and validation set
        L_valid(e) = out{KKK(ii,e)}.L_valid;
        % Select the accuracy on teh validation set
        L_test(e) = out{KKK(ii,e)}.L_test;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Select the value of parameters
        N_Markov(e) = out{KKK(ii,e)}.N_params;
        % Select the value of gamma in the training set
        gamma_train{e} = out{KKK(ii,e)}.gamma_train;
        % Select the value of gamma in the validation set
        gamma_valid{e} = out{KKK(ii,e)}.gamma_valid;
        % Select the value of gamma in the test set
        gamma_test{e} = out{KKK(ii,e)}.gamma_test;
        % Select Lambda   % Why is called P?
        Lambda_fit{e} = out{KKK(ii,e)}.Lambda;
        % Select W
        W_fin{e} = out{KKK(ii,e)}.W';
        % Select S
        C_fin{e} = out{KKK(ii,e)}.C;
        
    end

    % Output the best prediction based on the annealing steps
    % and the combination of regularization parameters
    [~,e] = min(lll);
    out_opd.gamma_train = gamma_train{e};
    out_opd.gamma_test = gamma_test{e};
    out_opd.gamma_valid = gamma_valid{e};
    out_opd.C = C_fin{e};
    out_opd.L_train = L_fin(e);
    out_opd.L_valid = L_valid(e);
    out_opd.L_test = L_test(e);
    out_opd.reg_param_W = reg_param(1,e);
    out_opd.reg_param_CL = reg_param(2,e);
    out_opd.reg_param_L2 = reg_param(3,e);
    out_opd.W = W_fin{e};
    out_opd.Lambda = Lambda_fit{e};
    out_opd.time_SPARTA = mean(mean(time));
    out_opd.N_params_Markov = N_Markov(e);

    % new variable for W time
    out_opd.totalTimeW = totalTimeW;

end




