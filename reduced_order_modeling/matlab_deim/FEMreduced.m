function [Ah,Fh] = FEMreduced(Gam,mesh,rElem,rSeg)
% This function will take the configuration and a triangulation and return Aq and Fh
% by using finite element method
global GeometryOrder SolutionOrder
global nshapeSolution nshapeSolutionLine
global nsize
global nelemNode nseg nsegNode nsegRhs
global data

%%%%%   INTERIOR ELEMENT CONTRIBUTION

nzmax = 6*nsize;
cols = zeros(1,nzmax);
rows = cols;
s = zeros(1,nzmax);
nentries = 0;

Node = zeros(nelemNode,1);

% Loop over all elements
for ie = rElem'
    
  % Take global node for current element
  Node(1:nelemNode) = mesh.ElementGroup(ie, 1:nelemNode);
  
  %% Compute the element matrix using quadrature rule
  %% This is actually the Galerkin term
  gam = Gam(Node);
  [AI]=ElementStiffness(ie,gam);
  
  % Assembly matrix A
  
  for alpha = 1:nshapeSolution
    i = Node(alpha);
    for beta = 1:nshapeSolution
      j = Node(beta);
      
      nentries = nentries + 1;
      if (nentries > nzmax)
        nzmax = 2*nzmax;
        cols(nzmax) = 0;
        rows(nzmax) = 0;
        s(nzmax) = 0;
      end
      rows(nentries) = i;
      cols(nentries) = j;
      s(nentries) = AI(alpha, beta);
        
      %Ah(i,j) = Ah(i,j) + AI(alpha, beta);
    end
  end
end
    
Node = zeros(nsegNode,1);
    
for is = rSeg' %sum over boundary segs with reduced nodes
  
  Node(1:nsegNode) = mesh.BoundaryGroup(is, 1:nsegNode);
  
  % The boundary (traction boundary conditions)
  [AII] = Boundary2(SolutionOrder, GeometryOrder,is,2);
        
  % Assembly matrix A
  for alpha = 1:nshapeSolutionLine
    i = Node(alpha);
    for beta = 1:nshapeSolutionLine
      j = Node(beta);
  
      nentries = nentries + 1;
      if (nentries > nzmax)
        nzmax = 2*nzmax;
        cols(nzmax) = 0;
        rows(nzmax) = 0;
        s(nzmax) = 0;
      end
      rows(nentries) = i;
      cols(nentries) = j;
      s(nentries) = data.Bi*AII(alpha, beta);
      
%      Ah(i,j) = Ah(i,j) + data.Bi*AII(alpha, beta);
    end
  end
end
Ah = sparse(rows(1:nentries),cols(1:nentries),s(1:nentries),nsize,nsize);
%%%%%%%%%     RHS COMPUTATION

% Right hand side vector F
Fh = zeros(nsize,1);    

Node = zeros(nsegNode,1);

for is = 1:nsegRhs
  % Global node of current segment
  Node(1:nsegNode) = mesh.RHS(is, 1:nsegNode);
  
  % Compute the RHS
  [Fe] = Boundary2(SolutionOrder, GeometryOrder,is,1);
  
  % NEED TO CHANGE HERE FOR HIGHER ORDER SOLUTION
  
  for alpha = 1:nshapeSolutionLine
    i = Node(alpha);
    Fh(i) = Fh(i) + Fe(alpha);
  end
    
end


