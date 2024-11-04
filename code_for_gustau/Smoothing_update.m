function [zk_smooth_new,Pk_smooth_new,Gk] = Smoothing_update(zk_kal,Pk_kal,zk_smooth_past,Pk_smooth_past,D1,D2,R,Q)

zk_minus = D1*zk_kal;
Pk_minus = D1*Pk_kal*D1' + Q;

Gk = Pk_kal*D1'*pinv(Pk_minus);
zk_smooth_new = zk_kal + Gk*(zk_smooth_past - zk_minus);
Pk_smooth_new = Pk_kal + Gk*(Pk_smooth_past - Pk_minus)*Gk';
 
