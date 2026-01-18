function [nodeIDs, edgeList] = generateSteinerTopology(nTerm)
% Full Steiner tree: terminals degree-1, Steiners degree-3, connected & acyclic.
% Terminals  : IDs = -1 ... -nTerm
% Steiners   : IDs =  1 ...  (nTerm-2)

P = nTerm - 2;
if nTerm < 2, error('Need at least 2 terminals'); end
if nTerm == 2
    nodeIDs  = [-1 -2];          % trivial two-terminal case
    edgeList = [-1 -2];
    return
end

% 3 terminals is base case that is built into a larger tree
if nTerm == 3
    edgeList = [ 1  -1;
             1  -2;
             1  -3 ];
    nodeIDs = [-1, -2, -3, 1];
    return
end

% --- Base tree: one Steiner (ID=1) connects first 3 terminals
edgeList = [ 1  -1;
             1  -2;
             1  -3 ];
nextSteiner  = 2;      % next unused positive ID
nextTerminal = -4;     % next unused negative ID
edgePtr      = 4;      % next free row index (we preallocate below)

% total rows needed = 2*nTerm - 3
edgeList(2*nTerm-3, :) = 0;   % preallocate with zeros

% --- Iteratively add each remaining terminal
while abs(nextTerminal) <= nTerm
    % choose a random edge to split
    idx = randi(edgePtr-1);           % 1 .. current edge count
    U   = edgeList(idx,1);
    V   = edgeList(idx,2);

    S   = nextSteiner;  nextSteiner  = nextSteiner + 1;
    T   = nextTerminal; nextTerminal = nextTerminal - 1;

    % overwrite the chosen edge with terminal-Steiner edge
    edgeList(idx,:)   = [U  S];       % (U , new terminal)

    % add two new edges
    edgeList(edgePtr,:)   = [S V];    edgePtr = edgePtr + 1;
    edgeList(edgePtr,:)   = [S T];    edgePtr = edgePtr + 1;
end
% trims preallocated matrix to size
edgeList = edgeList(1:edgePtr-1,:);

% counts from -1 up to -nTerm, then from 1 to P.
% used for indexing into the coordinate list later
nodeIDs  = [-1:-1:-nTerm, 1:P];
end