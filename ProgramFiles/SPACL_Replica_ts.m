%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% eSPA main algorithm with a triple split (into training, validation and testing data)
%%
%%
%% SPARTAn is (c) 2022, Illia Horenko. SPARTAn is published and distributed under the Academic Software License v1.0 (ASL). SPARTAn is distributed in the hope
%% that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
%% A PARTICULAR PURPOSE. See the ASL for more details. You should have received a copy of the ASL along with this program; if not, write to horenkoi@usi.ch
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input - in: structure containing all the values computed in "SPACL_kmeans_dim_entropy_analytic_v3"


function [out] = SPACL_Replica_ts(in)    
    
    % Importing all the relevant parameters from 'SPACL_kmeans_dim_entropy_analytic_v3'
    eps_C = in.reg_param(1);
    X = in.X;
    N_neurons = in.N_neurons;
    gamma = in.gamma;
    C = in.C;
    W = in.W;
    %% Update for "triple split"
    T_v=size(in.X_valid,2);
    T_vh=floor(size(in.X_valid,2)/2);
    X_valid = in.X_valid(:,1:T_vh);
    pi_valid = in.pi_valid(:,1:T_vh);
    X_valid_ts = in.X_valid(:,(1):T_v);
    pi_valid_ts = in.pi_valid(:,(1):T_v);
    %%%%%%%%%%%%%%%%%%%%%%%%
    T = in.T;
    K = in.K;
    N_anneal = in.N_anneal;
    Pi = in.Pi;
    flag_AUC = in.flag_AUC;
    Lambda = in.Lambda;
    m = size(Pi,1);
    d = in.d;
    reg_param = in.reg_param(2);
    i = 1;
    delta_L = 1e10; eps = 1e-8;           
    eps_Creg = 1e-10;            
    MaxIter = 300;
    L = [];
    tic;
   
    timeW = 0;


    % Main loop for the computation of the four steps (p. 1572 eSPA paper)
    while and(delta_L > eps, i <= MaxIter)  % Stop criterion
    
        % Initialization of the computation of the Lambda-step       
        if i == 1
            W_m = sqrt(W)';
            % Pre-multiplication of W with X, analytic solution eSPA paper p.1567
            X_W = bsxfun(@times,X,W_m);  % element-wise multiplication              
        end                

        % Pre-multiplication of W with C, analytic solution (eSPA paper p.1567)
        C_W = bsxfun(@times,C,W_m);     % element-wise multiplication

        % Evaluation of the gamma step through analytical solution (eSPA paper p.1567)
        [gamma] = SPACL_EvaluateGamma(X_W, Pi, C_W, Lambda, T, K, m, d, reg_param);
        not_empty = sum(gamma,2) > 0;
        if sum(not_empty) ~= size(gamma,1)
            gamma = gamma(not_empty,:);
            C = C(:,not_empty);
            %Lambda=Lambda(:,not_empty);
            clear C_W; K = sum(not_empty);
        end
        % Attempting to measure W time
        time_W = tic();

        % Computation of the W -----------> ask Prof. Horenko: is this an analytical solution?
        [W] = SPACL_dim_entropy_EvaluateWRegularize_v3(X, gamma, C, d, T, W, eps_C);

        timeW = timeW + toc(time_W);

        % Evaluation of the S-step         
        [C] = SPACL_EvaluateCRegularize_analytic(X, gamma, K, d, T); 

        % Evaluation of the Lambda step through analytical solution (eSPA paper p.1567)
        [Lambda] = SPACL_EvaluateLambdaRegularize(Pi, gamma, m, K);

        

        % ___________________________________________Four steps finished________________________________________________

        % Recompute the updated values of X and C
        W_m = sqrt(W)';
        X_W = bsxfun(@times,X,W_m);     
        C_W = bsxfun(@times,C,W_m);    

        % Compute the value of the functional L according to eq. 2.4 p. 1571 eSPA paper
        [L_3] = SPACL_dim_entropy_L(X_W, Pi, C_W, Lambda, gamma, T, d, m, reg_param, eps_C, W, K, eps_Creg);
        L = [L L_3];
        
        % Compute the delta of the function for the tolerance condition
        if i > 1 
            delta_L = (L(i-1) - L(i));
        end

        % Update the iteration index
        i = i+1;

    end % end of the main While

    out.time = toc;
    T_valid = size(X_valid,2);
    P = Lambda;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    

    % changing the computation leads to a massive performance boost

    [gamma2] = SPACL_EvaluateGamma_valid(sqrt(W').*X,C_W,T,K);                 

    % Computing the accuracy in the training set 
    [L_pred_Markov_train2] = AUC_of_Prediction(gamma2,P,Pi,0,flag_AUC);  
    L_pred_Markov_train2 = L_pred_Markov_train2 / (T*m);

    % Probably never used, why not an error message if K<2 -- i.e., K=1?
   % if K < 2 
   %     [gamma1,LLL2] = EvaluateGammaParallel_vect(diag(sqrt(W))*X, C_W, T, d, K, []);        
   % else
        gamma1 = gamma2;    
   % end
    [P] = SPACL_EvaluateLambdaRegularize(Pi, gamma1, m, K);
    % Recompute the accuracy in the training set (is different from the previous one iff K<2) 
    [L_pred_Markov_train1] = AUC_of_Prediction(gamma1, P, Pi, 0, flag_AUC);         
    L_pred_Markov_train1 = L_pred_Markov_train1 / (T*m);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the actual number of discrete boxes K (i.e. count the boxes in which there is at least one observation)
    K_actual = length(find(sum(gamma') > 1e-7));
    out.N_params = d * (K_actual+1) + (m-1) * K_actual;

    % Computation of the gamma in the validation set
    %if L_pred_Markov_train1 < L_pred_Markov_train2
    %    [gamma_valid,LLL2] = EvaluateGammaParallel_vect(diag(sqrt(W))*X_valid,C_W,T_valid,d,K,[]);    
    %else
        [gamma_valid] = SPACL_EvaluateGamma_valid(sqrt(W').*X_valid,C_W,T_valid,K);
    %end

    % Compute the matrix Lambda for the validation set 
    gamma_valid = real(gamma_valid);
    [~,out.confusion_matr_valid,~,~] = confusion(pi_valid,P*gamma_valid);

    %[Lambda_valid,Lambda_Conf] = SPACL_EvaluateLambdaRegularize([Pi pi_valid],[gamma gamma_valid],m,K);
    [Lambda_valid] = SPACL_EvaluateLambdaRegularize([Pi pi_valid],[gamma gamma_valid],m,K);
    Lambda_Conf=[];

    % Compute the accuracy of the validation set
    [L_pred_Markov] = AUC_of_Prediction(gamma_valid, P, pi_valid, 0, flag_AUC);
    L_pred_Markov = L_pred_Markov / (T_valid*m);

    %% Addition for the "triple split"
    [gamma_valid_ts] = SPACL_EvaluateGamma_valid(sqrt(W').*X_valid_ts,C_W,T_v,K);
    gamma_valid_ts = real(gamma_valid_ts);
    [~,out.confusion_matr_valid_ts,~,~] = confusion(pi_valid_ts,P*gamma_valid_ts);
    [L_pred_Markov_ts] = AUC_of_Prediction(gamma_valid_ts, P, pi_valid_ts, 0, flag_AUC);
    L_pred_Markov_ts = L_pred_Markov_ts / ((T_v)*m);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Computing the first part of the functional L
    % for each combination of regularization parameters

    % In the training set
    L_discr_train = 0;
    for t = 1:T
        dist = X(:,t) - C*gamma(:,t);
        L_discr_train = L_discr_train + dist'*dist;
    end
    L_discr_train = L_discr_train/(T*d);

    % In the validation set
    L_discr_valid = 0;
    for t = 1:T_valid
        dist = X_valid(:,t) - C*gamma_valid(:,t);
        L_discr_valid = L_discr_valid + dist'*dist;
    end
    L_discr_valid = L_discr_valid / (T_valid*d);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Saving all the relevant variables
    out.L = L_3;
    out.L_discr_train = L_discr_train;
    out.L_discr_valid = L_discr_valid;
    out.L_pred_Markov = L_pred_Markov;
    %% Addition for the "triple split"
    out.L_pred_Markov_ts = L_pred_Markov_ts;
    out.P = P;
    out.P_valid = Lambda_valid;
    out.P_valid_conf = Lambda_Conf;
    out.W = W;
    out.gamma = gamma;
    out.gamma_valid = gamma_valid;
    out.C = C;
    out.reg_param = reg_param;

    % new output for time W
    out.timeW = timeW;
end