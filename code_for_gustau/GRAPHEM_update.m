function D1 = GRAPHEM_update(Sigma,Phi,C,K,sigma_Q,reg,D10,Maj_D1)

%M-step of EM algorithm in GRAPHEM algorithm
 
reg1 = reg.reg1;%(1);
gamma1 = reg.gamma1;

ItDR = 10000;
display = 0;
precision = 1e-3;
 

switch reg1
    case 0 %no penalization
        D1 = C*inv(1/2 * (Phi + Phi'));
    case 2 %l2 penalization
        temp = K/sigma_Q^2;
        D1 = temp*C*inv(Phi*temp + gamma1*eye(size(Phi,1)));
    case 1 %l1 penalization
        %run Douglas rachford algorithm 
        Y = D10;%initialization
        %disp(Maj_D1)
        for i = 1:ItDR
            D1 = prox_L1(1,gamma1,Y); % prox l1 at A
            V  = prox_ML_D1(C,Phi,sigma_Q,1,2*D1-Y,K); % prox f1 at 2A - Y (f1 == q)
            Y  = Y + V - D1;
            
            obj(i) = ComputeMaj_D1(sigma_Q,D1,Sigma,Phi,C,K) ...
                + gamma1 * sum(abs(D1(:))); % q + L0 = Q
            if(i>1)
                if(abs(obj(i)-obj(i-1))<=precision && obj(i) < Maj_D1)
                    %disp(['DR stops at i = ',num2str(i)]);
                    break;
                end
            end
            if(display)
            disp(['i = ',num2str(i),'; obj = ',num2str(obj(i))]);
            end
            
        end
end


end



