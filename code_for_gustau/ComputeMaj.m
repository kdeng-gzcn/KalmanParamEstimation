function Maj = ComputeMaj(z0,P0,Q,R,z_mean_smooth0,P_smooth0,D1,D2,Sigma,Phi,B,C,D,K)

Maj1 = 1/2 * log(det(2*pi*P0)) + K/2 * log(det(2*pi*Q)) + K/2 * log(det(2*pi*R));
Maj2 = 1/2 * trace(inv(P0)*(P_smooth0 + (z_mean_smooth0-z0)*(z_mean_smooth0-z0)'));
% maj3 is the majorant in paper
Maj3 = K/2 * trace(inv(Q)*(Sigma - C*D1'-D1*C' + D1*Phi*D1'));
Maj4 = K/2 * trace(inv(R)*(D - B*D2'-D2*B' + D2*Sigma*D2'));

Maj = Maj1 + Maj2 + Maj3 + Maj4;





