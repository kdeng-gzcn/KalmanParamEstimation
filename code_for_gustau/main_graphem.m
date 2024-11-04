close all
clear all
clc

%% generate synthetic data

K  = 1000; % lenght of time series
flag_plot = 1;

load datasetC.mat % load data

Nx = size(D1,1); % number of nodes
Nz = Nx;
D2 = eye(Nz);   % for simplicity and identifiability purporses 

sigma_Q = 0.1; % observation noise std
Q = sigma_Q^2*eye(Nz);
sigma_R = 0.1; % state noise std
R = sigma_R^2*eye(Nx);
sigma_P = 0.0001; % prior state std
P0 = sigma_P^2*eye(Nz);
z0 = rand(Nz,1);

reg1 = 1; gamma1 = 5e1; % regularizer

reg.reg1 = reg1;
reg.gamma1 = gamma1;


%%
Nreal = 10; % Number of independent runs
for real = 1:Nreal

    disp(['---- REALIZATION ',num2str(real),' ----']);
    


%% Model generation
z = zeros(Nz,K);
x = zeros(Nx,K);
z(:,1) = D1*(z0 + sigma_P * randn(Nz,1)) + sigma_Q * randn(Nz,1);%initialization
x(:,1) = D2*z(:,1) + sigma_R * randn(Nx,1);    
for k = 2:K
    z(:,k) = D1*z(:,k-1) +  sigma_Q  * randn(Nz,1);
    x(:,k) = D2*z(:,k)   +  sigma_R  * randn(Nx,1);
end

% if(flag_plot==1)
%     figure(1)
%     subplot(121)
%     plot(x')
%     xlabel('k')
%     title('observed sequence x')
%     subplot(122)
%     plot(z')
%     xlabel('k')
%     title('unknown state z')
% end

saveX(:,:,real) = x;
 
%% Inference (GRAPHEM algorithm)
disp('-- GRAPHEM --')
disp(['Regularization on D1: norm ',num2str(reg1),' with gamma1 = ',num2str(gamma1)]);

Err_D1 = [];
Nit_em = 50; %number of iterations maximum for EM loop
prec = 1e-3; %precision for EM loop

%initialization of GRAPHEM
D1_em = prox_stable(CreateAdjacencyAR1(Nz,0.1),0.99);
D1_em_save = [];

for i = 1:Nit_em % EM iterations
    %disp(['EM iteration ',num2str(i)]);
    
    %1/ Kalman filter filter
    z_mean_kalman_em = zeros(Nz,K);
    P_kalman_em = zeros(Nz,Nz,K);
    yk_kalman_em = zeros(Nx,K);
    Sk_kalman_em = zeros(Nx,Nx,K);    
    [z_mean_kalman_em(:,1),P_kalman_em(:,:,1),yk_kalman_em(:,1),Sk_kalman_em(:,:,1)] = Kalman_update(x(:,1),z0,P0,D1_em,D2,R,Q);
    for k = 2:K
        [z_mean_kalman_em(:,k),P_kalman_em(:,:,k),yk_kalman_em(:,k),Sk_kalman_em(:,:,k)] = Kalman_update(x(:,k),z_mean_kalman_em(:,k-1),P_kalman_em(:,:,k-1),D1_em,D2,R,Q);
    end
    
    %compute loss function (ML for now, no prior)
    PhiK(i) = Compute_PhiK(0,Sk_kalman_em,yk_kalman_em);
    
    %compute penalty function  before update
    Reg_before = Compute_Prior_D1(D1_em,reg);
    MLsave(i) = PhiK(i);
    Regsave(i) = Reg_before;
    PhiK(i) = PhiK(i) + Reg_before; %update loss function   


    %2/ Kalman smoother
    z_mean_smooth_em = zeros(Nz,K);
    P_smooth_em = zeros(Nz,Nz,K);
    G_smooth_em = zeros(Nz,Nz,K);
    z_mean_smooth_em(:,K) = z_mean_kalman_em(:,K);
    P_smooth_em(:,:,K) = P_kalman_em(:,:,K);
    for k = fliplr(1:K-1)
        [z_mean_smooth_em(:,k),P_smooth_em(:,:,k),G_smooth_em(:,:,k)] = ...
            Smoothing_update(z_mean_kalman_em(:,k),P_kalman_em(:,:,k),z_mean_smooth_em(:,k+1),P_smooth_em(:,:,k+1),D1_em,D2,R,Q);
    end
    [z_mean_smooth0_em,P_smooth0_em,G_smooth0_em] = Smoothing_update(z0,P0,z_mean_smooth_em(:,1),P_smooth_em(:,:,1),D1_em,D2,R,Q);

    % compute EM parameters
    [Sigma,Phi,B,C,D] = EM_parameters(x,z_mean_smooth_em,P_smooth_em,G_smooth_em,z_mean_smooth0_em,P_smooth0_em,G_smooth0_em);
    
    %compute majorant function for ML term before update
    Maj_before(i) = ComputeMaj(z0,P0,Q,R,z_mean_smooth0_em,P_smooth0_em,D1_em,D2,Sigma,Phi,B,C,D,K);
    
    Maj_before(i) = Maj_before(i) + Reg_before;     %add prior term (= majorant for MAP term)
    
    %3/ EM Update
    Maj_D1_before = ComputeMaj_D1(sigma_Q,D1_em,Sigma,Phi,C,K) + Reg_before;
    D1_em_ = GRAPHEM_update(Sigma,Phi,C,K,sigma_Q,reg,D1_em,Maj_D1_before);
    
    %compute majorant function for ML term after update (to check decrease)
    Maj_after(i) = ComputeMaj(z0,P0,Q,R,z_mean_smooth0_em,P_smooth0_em,D1_em_,D2,Sigma,Phi,B,C,D,K);
    %add penalty function after update
    Reg_after  = Compute_Prior_D1(D1_em_,reg);
    Maj_after(i) = Maj_after(i) + Reg_after;
    
    
    if(Maj_after(i)> Maj_before(i))
        disp(['ERROR: Majorant increases at iteration ',num2str(i)]);
        disp(Maj_after(i) - Maj_before(i))
    end
    
    D1_em = D1_em_; % D1 estimate updated (which will be used in the next iteration for Kalman)
    
    D1_em_save(:,:,i) = D1_em; % keep track of the sequence
    
    Err_D1(i) = norm(D1-D1_em,'fro')/norm(D1,'fro');  % 
    
        if(i>1)
        if(abs(PhiK(i)-PhiK(i-1))< prec)
            disp(['EM converged after iteration ',num2str(i)]);
            break;
        end
        end
    
end

% validation in terms of true positive/false positive/error in D1, etc. 

D1_em_save = squeeze(D1_em_save);
disp(['Final error on D1 = ',num2str(Err_D1(end))]);

threshold = 1e-10;
D1_binary = abs(D1)>=threshold;
D1_em_binary = abs(D1_em)>=threshold;
[TP, FP, TN, FN] = calError(D1_binary, D1_em_binary);

figure(30)
subplot(121)
G = digraph(D1);
LWidths = 5*abs(G.Edges.Weight)/max(G.Edges.Weight);
hfig = plot(G,'LineWidth',LWidths);
subplot(122)
G_em = digraph(D1_em);
LWidths_em = 5*abs(G_em.Edges.Weight)/max(G.Edges.Weight);
plot(G_em,'XData',hfig.XData,'YData',hfig.YData,'LineWidth',LWidths_em);
set(gca,'xtick',[])
set(gca,'ytick',[])

precision(real) = TP/(TP + FP);
recall(real) = TP/(TP + FN);
specificity(real) = TN/(TN + FP);
accuracy(real) = (TP + TN)/(TP+TN+FP+FN);
RMSE(real) = Err_D1(end);
F1score(real) = 2*TP /(2*TP + FP + FN);
%disp(['TP = ',num2str(TP),'; TN = ',num2str(TN),'; FP = ',num2str(FP),'; FN = ',num2str(FN)]);
disp(['accuracy = ',num2str(accuracy(real)),'; precision = ',num2str(precision(real)),'; recall = ',num2str(recall(real)),'; specificity = ',num2str(specificity(real))]);


end

disp(['average RMSE = ',num2str(mean(RMSE))])
disp(['average accuracy = ',num2str(mean(accuracy))])
disp(['average precision = ',num2str(mean(precision))])
disp(['average recall = ',num2str(mean(recall))])
disp(['average specificity = ',num2str(mean(specificity))])
disp(['average F1 score = ',num2str(mean(F1score))])

%%
if(flag_plot==1)
    
%     figure(2)
%     plot(z')
%     hold on
%     plot(z_mean_kalman_em','--')
%     hold on
%     plot(z_mean_smooth_em',':')
%     legend('original','filter','smooth')
%     xlabel('k')
%     ylabel('z')
%     title('Results for Kalman with estimated D1')

    
    figure(3)
    semilogy(Err_D1)
    title('Error on D1')
    xlabel('GRAPHEM iterations')
    
    
    figure(4)
    plot(PhiK)
    title('Loss function')
    xlabel('GRAPHEM iterations')
 
end