function [J,G] = TMRThermalFinCost(Gam, Method)

global mesh data it L U P Q Fh uh phi costGrad prior Utilde Wtilde costh


%if(strcmp(Method, 'StateParameter') == 1 || strcmp(Method,
%'ParameterState') == 1 || strcmp(Method, 'Parameter') == 1) %'NoReduction'

switch Method
    case {'StateParameter','ParameterState','Parameter'}
    Gam1 = Gam;
    Gam = Wtilde*Gam;
end


% forward solve
%if(strcmp(Method, 'StateParameter') == 1 || strcmp(Method, 'ParameterState') == 1)
if(strcmp(Method, 'StateParameter') == 1 || strcmp(Method, 'ParameterState') == 1 || strcmp(Method, 'State') == 1)
    [Ah,Fh] = FEMsparse(Gam,mesh);
    %keyboard
    load MDEIMdata
    [K,~] = FEMreduced(Gam,mesh,rElem,rSeg);
    k = reshape(K,[length(Fh)^2,1]);
    k = k(sub2ind(size(K),rInd,cInd));
    kI = k(Ind');
    phiI = phi(Ind,:);
    %keyboard
    theta = phiI\kI;
    As = zeros(100,100); % fix this later...
    for i=1:length(theta)
       As =  As + theta(i)*Ai{1,i};
    end
    Fs = Utilde.'*Fh;   
    us = As\Fs;
    uh = Utilde*us;
    %keyboard
else
    [Ah,Fh] = FEMsparse(Gam,mesh);
    uh = Ah\Fh;
end

[L,U,P,Q] = lu(Ah);
% The exact QoI
%phi = uh(data.B);
Hobs = zeros(length(data.B), length(uh));
Hobs(1:length(data.B), data.B) = eye(length(data.B));
Hobs = sparse(Hobs);
phi = Hobs*uh;

% $$$ 
% $$$ % data fusion
% $$$ epsTsigmaInvh = data.epsTsigmaInvh;
% $$$ sigmaInvh     = data.sigmaInvh; % inverse half of observation covariance
% $$$ Ns            = data.Ns;
% $$$ 

% compute the cost function
% $$$ 
% $$$ costh = epsTsigmaInvh * (phi - data.phi_obs);
% $$$ 
% $$$   J1 = 0.5 * sum(costh.^2) / Ns;
% $$$   


J1 = 0.5 / data.sigma2 * sum((phi-data.phi_obs).^2);

% add regularization
prior = (data.PhiM * (Gam-data.Gam0)) ./ data.lam;
J2 = 0.5*data.alpha * sum(prior.^2);
data.J = J1+J2;


J = data.J;



data.ForwardSolution = uh;
if(strcmp(Method, 'StateParameter') == 1 || strcmp(Method, 'ParameterState') == 1 || strcmp(Method, 'State') == 1)
    data.Stiffness = As;
end
data.L = L; data.U = U; data.P = P; data.Q = Q;
data.ForwardRHS = Fh;
data.phi = phi;

% Solve the adjoint problem
%AdjointRHS = zeros(size(Fh));

%AdjointRHS(data.B) = Utilde*(-1/data.sigma2*(Utilde'*(data.phi - data.phi_obs)));
if(strcmp(Method, 'StateParameter') == 1 || strcmp(Method, 'ParameterState') == 1 || strcmp(Method, 'State') == 1)
    AdjointRHS = Utilde.'*Hobs.'*(-1/data.sigma2*(data.phi - Hobs*Utilde*((Utilde.'*Hobs')*data.phi_obs)));
    data.AdjointSolution = Utilde*(As.'\AdjointRHS);
else
    AdjointRHS = Hobs.'*(-1/data.sigma2*(Hobs*data.phi - data.phi_obs));
    data.AdjointSolution = (Ah.'\AdjointRHS);
end



G = StiffnessGradientPoint(Gam);

switch Method
  case {'StateParameter','ParameterState','Parameter'}
    G = Wtilde.'*(G + data.alpha  * data.PhiM.' * (prior ./ ...
                                                   data.lam));
  otherwise
    %add regularization part
    G = G + data.alpha  * data.PhiM.' * (prior ./ data.lam);
end
costGrad = true;

gradVerify = 0;
if (gradVerify == 1)
    Gamma = Gam;
    gDiff = G;
    GammaPert = Gam;
    epsilon = 1e-6;
    if(strcmp(Method, 'StateParameter') == 1 || strcmp(Method, 'ParameterState') == 1)
        Gamma = Gam1;
        GammaPert = Gam1;
        
        for i = 1:length(Gamma)
            GammaPert(i) = Gamma(i) + sqrt(-1)*epsilon;
            Gam = Wtilde*GammaPert;
            [Ah,Fh] = FEMsparse(Gam,mesh);
            As = transpose(Utilde)*(Ah*Utilde);
            Fs = transpose(Utilde)*Fh;
            us = As\Fs;
            uh = Utilde*us;
            phi = uh(data.B);
            JPert = 0.5 / data.sigma2 * sum((phi-data.phi_obs).^2);
            prior = (data.PhiM * (Gam-data.Gam0)) ./ data.lam;
            JPert = JPert + 0.5*data.alpha * sum(prior.^2);
            gDiff(i) = imag(JPert)/epsilon;
            GammaPert(i) = Gamma(i);
        end
    else
        
        for i = 1:length(Gamma)
            GammaPert(i) = Gamma(i) + sqrt(-1)*epsilon;
            Gam = GammaPert;
            [Ah,Fh] = FEMsparse(Gam,mesh);
            As = transpose(Utilde)*(Ah*Utilde);
            Fs = transpose(Utilde)*Fh;
            us = As\Fs;
            uh = Utilde*us;
            phi = uh(data.B);
            JPert = 0.5 / data.sigma2 * sum((phi-data.phi_obs).^2);
            prior = (data.PhiM * (Gam-data.Gam0)) ./ data.lam;
            JPert = JPert + 0.5*data.alpha * sum(prior.^2);
            gDiff(i) = imag(JPert)/epsilon;
            GammaPert(i) = Gamma(i);
        end
    end
    GD = abs((gDiff-G)./gDiff);
    keyboard
    
end

