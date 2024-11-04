function [Sigma,Phi,B,C,D] = EM_parameters(x,z_mean_smooth,P_smooth,G_smooth,z_mean_smooth0,P_smooth0,G_smooth0)

K = size(x,2);
Nx = size(x,1);
Nz = size(z_mean_smooth,1);

%initialize
Sigma = zeros(Nz,Nz);
Phi   = zeros(Nz,Nz);
B     = zeros(Nx,Nz);
C     = zeros(Nz,Nz);
D     = zeros(Nx,Nx);

for k = 2:K
    Sigma = Sigma + 1/K*(P_smooth(:,:,k) + z_mean_smooth(:,k)*z_mean_smooth(:,k)');
    Phi = Phi + 1/K * (P_smooth(:,:,k-1) + z_mean_smooth(:,k-1)*z_mean_smooth(:,k-1)');
    B   = B + 1/K*(x(:,k)*z_mean_smooth(:,k)');
    C   = C + 1/K*(P_smooth(:,:,k)*G_smooth(:,:,k-1)' + z_mean_smooth(:,k)*z_mean_smooth(:,k-1)');
    D   = D + 1/K*(x(:,k)*x(:,k)');
end
%for k = 1
Sigma = Sigma + 1/K*(P_smooth(:,:,1) + z_mean_smooth(:,1)*z_mean_smooth(:,1)');
Phi   = Phi + 1/K * (P_smooth0 + z_mean_smooth0*z_mean_smooth0');
B     = B + 1/K*(x(:,1)*z_mean_smooth(:,1)');
C     = C + 1/K*(P_smooth(:,:,1)*G_smooth0' + z_mean_smooth(:,1)*z_mean_smooth0');
D     = D + 1/K*(x(:,1)*x(:,1)');
