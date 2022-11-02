%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% eSPA main algorithm with a triple split (into training, validation and testing data)
%%
%% (c) Illia Horenko 2022, GNU General Public License v2.0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input - in: structure containing all the values computed in "SPACL_kmeans_dim_entropy_analytic_v3"


function [out] = SPARTA_Replica_ts(in)    
    
    % Importing all the relevant parameters from 'SPACL_kmeans_dim_entropy_analytic_v3'
    eps_W = in.reg_param(1);
    eps_CL=in.reg_param(2);
    eps_L2=in.reg_param(3);
    X = in.X;
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
    Pi = in.Pi;
    Lambda = in.Lambda;
    m = size(Pi,1);
    d = in.d;
    i = 1;
    delta_L = 1e10; eps = 1e-10;           
    MaxIter = 300;
    L = [];
    tic;
   
    timeW = 0;

%% TODO:sparse gamma?
    % Main loop for the computation of the four steps (p. 1572 eSPA paper)
    while and(delta_L > eps, i <= MaxIter)  % Stop criterion
    

        % Evaluation of the gamma step through analytical solution
        [gamma] = SPARTA_EvaluateGamma(X, Pi, C, Lambda, T, K, m, d, eps_CL,W);
        %[L_3] = SPARTA_L(X,Pi,C,Lambda,gamma,T,d,m, eps_CL, eps_W,W,K,eps_L2)
        % Eliminate empty clusters
        not_empty = sum(gamma,2) > 0;
        if and(sum(not_empty) ~= size(gamma,1),size(gamma,1)>1)
            gamma = gamma(not_empty,:);
            C = C(:,not_empty);
            Lambda=Lambda(:,:,not_empty);
            clear C_W; K = sum(not_empty);
        end
        % Attempting to measure W time
        time_W = tic();

        % Computation of the W
        [W] = SPARTA_EvaluateW(X, Pi, gamma, C, Lambda, d, T,K,m, W, eps_W,eps_CL);
        %WWW(i,:)=W;
        %[L_3] = SPARTA_L(X,Pi,C,Lambda,gamma,T,d,m, eps_CL, eps_W,W,K,eps_L2)

        timeW = timeW + toc(time_W);

        % Evaluation of the C-step         
        [C] = SPARTA_EvaluateC(X,gamma,K,d,T);
        %[L_3] = SPARTA_L(X,Pi,C,Lambda,gamma,T,d,m, eps_CL, eps_W,W,K,eps_L2)

        % Evaluation of the Lambda step 
        %[Lambda]=SPARTA_EvaluateLambdaRegularize(X,W,Pi,gamma,m,K,d,T,eps_CL,eps_L2); 
        [Lambda]=SPARTA_EvaluateLambdaRegularize_v5(Lambda,X,W,Pi,gamma,m,K,d,T,eps_CL,eps_L2); 
        

        % ___________________________________________Four steps finished________________________________________________

        % Compute the value of the functional L according to eq. 2.4 p. 1571 eSPA paper
        [L_3] = SPARTA_L(X,Pi,C,Lambda,gamma,T,d,m, eps_CL, eps_W,W,K,eps_L2);
        L = [L L_3];
        
        % Compute the delta of the function for the tolerance condition
        if i > 1 
            delta_L = (L(i-1) - L(i));
        end
%          [gamma_train] = SPARTA_EvaluateGamma(X, Pi, C, Lambda, T, K, m, d, 0,W); 
%           figure(10);clf;subplot(2,2,1);
%           VisualizeBox(X,gamma_train);
%           subplot(2,2,2);plot(W,':o');
%           subplot(2,2,3);plot(L);
        % Update the iteration index
        i = i+1;
        pause(0.2);
    end % end of the main While

    out.time = toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    

    % SPARTA prediction error (in average squared l2-norm) for training data 
    [gamma_train] = SPARTA_EvaluateGamma(X, Pi, C, Lambda, T, K, m, d, 0,W); 
    [Lambda]=SPARTA_EvaluateLambdaRegularize(X,W,Pi,gamma_train,m,K,d,T,eps_CL,eps_L2); 
%     if isempty(gamma_train)
%        keyboard 
%     end
    [L_train,y_train] = SPARTA_pred_error(X,Pi,Lambda,gamma_train,T,m, W,K);
    % SPARTA prediction error (in average squared l2-norm) for validation data 
    [gamma_valid] = SPARTA_EvaluateGamma(X_valid, pi_valid, C, Lambda, T_vh, K, m, d, 0,W); 
    [L_valid,y_valid] = SPARTA_pred_error(X_valid,pi_valid,Lambda,gamma_valid,T_vh,m, W,K);
    % SPARTA prediction error (in average squared l2-norm) for validation data 
    [gamma_test] = SPARTA_EvaluateGamma(X_valid_ts, pi_valid_ts, C, Lambda, T_v, K, m, d, 0,W); 
    [L_test,y_test] = SPARTA_pred_error(X_valid_ts,pi_valid_ts,Lambda,gamma_test,T_v,m, W,K);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the actual number of discrete boxes K (i.e. count the boxes in which there is at least one observation)
    K_actual = length(find(sum(gamma') > 1e-7));
    out.N_params = d * (K_actual+1) + (m-1) * K_actual;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Saving all the relevant variables
    out.L = L_3;
    out.L_train = L_train;
    out.L_valid = L_valid;
    out.L_test = L_test;
    %% Addition for the "triple split"
    out.Lambda = Lambda;
    out.W = W;
    out.gamma_train = y_train;
    out.gamma_valid = y_valid;
    out.gamma_test = y_test;
    out.C = C;

    % new output for time W
    out.timeW = timeW;
end