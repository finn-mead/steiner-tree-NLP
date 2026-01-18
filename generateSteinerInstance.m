function [termXY, nodeIDs, edgeList, xGuess, bVal] = ...
           generateSteinerInstance(nTerm, scale, randFactorRange)
%------------------------------------------------------------------
% Generate a random instance of steiner tree
%   nTerm      - number of terminals
%   scale      - half-width of the square arena:
%                terminals lie in [-scale , scale] for each component
%   randFactorRange [a, b] denotes the multiplier used in generating bVal
%
% Returns
%   termXY   - nTerm x 2 terminal coordinates
%   nodeIDs  - [-1:-1:-nTerm  1:(nTerm-2)]   (terminal & Steiner IDs)
%   edgeList - (2*nTerm-3)x2 full-Steiner edge list  (degree T=1,  S=3)
%   xGuess   - 2Px1 initial Steiner guess  (P=nTerm-2)
%   bVal     - max allowed edge length
%------------------------------------------------------------------

    if nargin<3, randFactorRange = [0.5, 0.6]; end

    % 1. topology
    [nodeIDs, edgeList] = generateSteinerTopology(nTerm);
    P   = max(0, nTerm-2);

    % 2. terminals uniformly in the square
    termXY = -scale + 2*scale * rand(nTerm,2);

    % 3. initial Steiner guess:
    %    small random offsets then scaled
    % makes things faster by not having edge length constraints between
    % steiner points active at 0 (numerical issues) and also not too far
    % (more constraints initially violated)

    % normal distribution with std dev dispersion * scale
    initial_dispersion = 0.1;
    if P>0
        xGuess = (initial_dispersion*randn(2*P,1)) * scale;
    else
        xGuess = [];
    end

    % 4. Choose bVal from terminal geometry.
    % we dont actually put edges between terminals but taking the
    % max distance between any two terminals gives a heuristic for the
    % total spread, which we use to come up with b.
    Dmax = max(pdist(termXY));
    if Dmax==0, Dmax=1; end

    fmin = randFactorRange(1);
    fmax = randFactorRange(2);
    
    randF = fmin + (fmax-fmin)*rand;   % uniform in [fmin,fmax]
    bVal  = max(1.0, Dmax*randF);
end