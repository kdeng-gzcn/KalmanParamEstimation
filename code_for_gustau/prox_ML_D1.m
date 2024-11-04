function D1prox = prox_ML_D1(C,Phi,sigma_Q,gamma,D1,K)

temp = gamma*K/sigma_Q^2;

D1prox = (temp*C+D1)*inv(Phi*temp + eye(size(Phi,1)));