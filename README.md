## Description

This repository contains the data and code reproducing the comparison of NINO3.4 regression learning and prediction results (Fig. 1F-1K) from the paper:

Horenko, I., Vecchi, E., Kardos, J., Schenk, O. Wächter, A., O'Kane, T., Gagliardini, P. and Gerber, S. (2022). On cheap entropy-sparsified regression learning, under revision at the Proceedings of the National Academy of Sciences (PNAS).


## Meta-parameter ranges chosen for regression learners (SPARTA, DL LSTM, multilinear regression with elastic net regularization)

# SPARTA-metaparameters (to change, go to lines 77-84 in Demo_SPARTA_ENSO.m)

Number of SPARTA patterns/boxes/clusters: K = 3;

The grid for SPARTA regularisation parameter \epsilon_w in front of the entropy-regularisation term:
reg_param_W_SPARTA=[1e-4 5e-4 1e-3 2e-3 4e-3 6e-3 8e-3 1e-2 0.05 10];

The grid for SPARTA regularisation parameter \epsilon_r in front of the regression learning term:
reg_param_CL_SPARTA=[0.1 5 30];

The grid for SPARTA regularisation parameter \epsilon_{l_2} in front of the l2-regularisation term:
reg_param_L_SPARTA=[5e-5 5e-4 1-4];

# DL LSTM metaparameters (to change, go to line 50 of ProgramFiles/DeepNN_Regression_ts.m)

Number of hidden neurons: N = [2 4 5 8 10 20 30 40 50];

# Multilinear regression with elastic net regularization (to change, go to line 190 in Demo_SPARTA_ENSO.m)

l1 regularisation parameter: [1e-9 1e-8 1e-7 1e-6 ... 1e2];
l2 regularisation parameter: [1e-9 1e-8 1e-7 1e-6 ... 1e2];
