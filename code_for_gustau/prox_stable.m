function Dprox = prox_stable(D,eta)

[U,S,V] = svd(D);
S = min(S,eta);
Dprox = U*S*V';