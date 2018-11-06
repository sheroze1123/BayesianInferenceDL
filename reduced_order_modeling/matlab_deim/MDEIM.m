% MDEIM.m
% computes offline data structures for MDEIM

% initialize mesh data
Globals; 
load coarseGrid
mesh = grid; clear grid
mesh.BoundaryGroup = mesh.BoundaryGroup{1};
mesh.RHS = mesh.RHS{1};
mesh.SolutionOrder = 1;
mesh.GeometryOrder = 1;
mesh.ElementGroup = [mesh.ElementGroup{1};mesh.ElementGroup{2}; ...
mesh.ElementGroup{3};...
mesh.ElementGroup{4}; mesh.ElementGroup{5}];
mesh.nElementGroup = 1;
GeometryOrder = mesh.GeometryOrder;
SolutionOrder = mesh.SolutionOrder;
I = 1:mesh.nodes;
data.B = I;
Nobs = length(I);
data.Nobs = Nobs;
data.al = 1.8;
data.Bi = 0.1;
data.alpha = 50;
ComputeGlobal;

% compute prior distribution parameters
[data.M,data.K] = ComputeMassStiff(mesh);
[data.V, Lam] = eig(full(data.K),full(data.M));
V = data.V;
data.V = V;
Lam = (diag(Lam)+1).^(data.al/2);
data.lam  = 1./Lam;
clear Lam
data.PhiM = data.V.' * data.M.';


% build snapshots of Ah by sampling from prior distribution
k=1000; % k is taken sufficiently large to capture all information about Ah
data.Gam0 = 0.;
for i=1:k
    data.Gam_obs = data.Gam0 +  data.V * (sqrt(1/data.alpha) * data.lam .* randn(size(mesh.coor,1),1));
    [Ah,~] = FEMsparse(data.Gam_obs,mesh);
    if i==1
        [rInd,cInd,Av]=find(Ah);
        X=zeros(length(Av),k);
    end
    X(:,i)=nonzeros(Ah);
end

% compute full and reduced basis for Ah
[W_full,~,~]=svd(X,'econ');
m=180; % 180 basis vectors captures 99% of the "energy" from these snapshots
phi = W_full(:,1:m);
Ind = DEIM(phi);
%Ind = sort(Ind); % maybe these shouldn't be sorted...

% form reduced basis matrices
Ai=cell(1,m);
load StateReduction
V = UU(:,1:100);
clear SS USnapshots UU VV
for i=1:m
    Ai{1,i} = V'*sparse(rInd,cInd,phi(:,i))*V; 
end

% detect reduced nodes
I=rInd(Ind');
J=cInd(Ind');
rNodes=unique([I',J']');

% detect reduced elements
rElem=[];
for i=1:length(mesh.ElementGroup)
    if sum(ismember(mesh.ElementGroup(i,:),rNodes)) > 0
        rElem = [rElem;i];
    end
end

% detect reduced boundary segments
rSeg=[];
for i=1:length(mesh.BoundaryGroup)
    if sum(ismember(mesh.BoundaryGroup(i,:),rNodes)) > 0
        rSeg = [rSeg;i];
    end
end

clearvars -except phi Ind rInd cInd rElem Ai rSeg