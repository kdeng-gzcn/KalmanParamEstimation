function Dprox = prox_L1(gamma,gammareg,D)

temp = gamma*gammareg;
Dprox = sign(D).*max(0,abs(D)-temp);