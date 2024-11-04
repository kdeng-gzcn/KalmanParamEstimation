function Reg1 = Compute_Prior_D1(D1,reg)

reg1 = reg.reg1;%(1);
gamma1 = reg.gamma1;

switch reg1
    case 0
        Reg1 = 0;
    case 1
        Reg1 = gamma1.*sum(abs(D1(:)));
    case 10
        Reg1 = gamma1.*sum(abs(D1(:)));
    case 2
        Reg1 = gamma1/2 * norm(D1,'fro')^2;
end

