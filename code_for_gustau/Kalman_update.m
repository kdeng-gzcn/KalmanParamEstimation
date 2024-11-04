function [zk_mean_new,Pk_new,yk,Sk] = Kalman_update(xk,zk_mean_past,Pk_past,D1,D2,R,Q)

zk_minus = D1*zk_mean_past;
Pk_minus = D1*Pk_past*D1' + Q;

yk = xk - D2*zk_minus;
Sk = D2*Pk_minus*D2' + R;
Kk = Pk_minus*D2'*pinv(Sk);
zk_mean_new = zk_minus + Kk*yk;
Pk_new = Pk_minus - Kk*Sk*Kk';

