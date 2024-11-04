function PhiK = Compute_PhiK(Phi0,Sk_kal,yk_kal)

K = size(Sk_kal,3);
PhiK = Phi0; %this would be our prior term

for k = 1:K
    PhiK = PhiK + 1/2 * log(det(2*pi*Sk_kal(:,:,k))) + ...
        1/2 * yk_kal(:,k)'*inv(Sk_kal(:,:,k))*yk_kal(:,k);
end


