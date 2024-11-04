function D1 = GRAPHEM_update_test_stability(Sigma,Phi,C,K,sigma_Q,reg,D10,Maj_D1)

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
            D1 = prox_L1(1,gamma1,Y);
            V  = prox_ML_D1(C,Phi,sigma_Q,1,2*D1-Y,K);
            Y  = Y + V - D1;
            
            obj(i) = ComputeMaj_D1(sigma_Q,D1,Sigma,Phi,C,K) ...
                + gamma1 * sum(abs(D1(:))); 
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
        
       case 10 %l1 penalization + stability
        %run PPXA algorithm
        W = [1,1,1]; W = W./sum(W);%consensus weights       
        Y1 = D10; Y2 = D10; Y3 = D10;
        D1 = W(1)*Y1 + W(2)*Y2 + W(3)*Y3;
        for i = 1:ItDR
            P1 = prox_L1(1/W(1),gamma1,Y1);
            P2 = prox_ML_D1(C,Phi,sigma_Q,1/W(2),Y2,K);
            P3 = prox_stable(Y3,0.99); 
            D1 = W(1)*P1 + W(2)*P2 + W(3)*P3;
            Y1 = Y1 + D1 - P1;
            Y2 = Y2 + D1 - P2;
            Y3 = Y3 + D1 - P3;
            obj(i) = ComputeMaj_D1(sigma_Q,D1,Sigma,Phi,C,K) ...
                + gamma1 * sum(abs(D1(:))); 
            %obj(i) = ComputeMaj_D1(sigma_Q,P1,Sigma,Phi,C,K) ...
            %    + gamma1 * sum(abs(P1(:))); 
            if(i>1)
                if(abs(obj(i)-obj(i-1))<=precision && norm(D1)< 1)
                %if(abs(obj(i)-obj(i-1))<=precision && norm(P1)< 1)
                    disp(['PPXA stops at i = ',num2str(i)]);
                    break;
                end
            end
            if(display)
            disp(['i = ',num2str(i),'; obj = ',num2str(obj(i))]);
            end
        end
        D1 = P1;
end


end



