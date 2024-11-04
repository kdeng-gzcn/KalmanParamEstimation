function Maj_D1 = ComputeMaj_D1(sigma_Q,D1,Sigma,Phi,C,K)

Maj_D1 = K/2 * trace(1/sigma_Q^2*(Sigma - C*D1'-D1*C' + D1*Phi*D1'));