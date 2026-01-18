% Functions to turn a given steiner tree into an NLP.
% creates:
%       objFun(x) - total squared edge length
%       constraintFunHandles{k}(x) - one g_k(x) per edge
%
% essentially this script just iterates over every edge
% for some topology and turns it into the NLP form we need.
% note that it does not convert it to a log barrier version yet.


function [objFun, constraintFuns] = steinerNLP(terminalCoords, ...
    edgeList, b, p)

    % 2.  Objective function lambda just depends on x, the steiner points in column vector form.
    % f(x) = sum of ||edge||^2
    objFun = @(xVec) totalSquaredEdgeLength(xVec, terminalCoords, edgeList, p);

    % 3.  Constraint function pointers.
    % g_k(x) = ||edge_k||^2 - b^2  (want <= 0)
    constraintFuns = buildEdgeConstraints( ...
        terminalCoords, edgeList, b, p);
end



function total = totalSquaredEdgeLength(xVec, terminalCoords, edges, p)
% xVec   -  2Px1 vector [S1x;S1y;S2x;S2y; ...]
% terminalCoords   -  Nx2 terminal coordinates
% edges  -  Mx2 list of endpoint IDs
% P      -  number of Steiner points

    steinerMat = vecToSteinerMatrix(xVec,p);   % Px2 matrix
    total = 0;
    for e = 1:size(edges,1)
        a = coordsOfID(edges(e,1), terminalCoords, steinerMat);
        b = coordsOfID(edges(e,2), terminalCoords, steinerMat);
        total = total + sum((a-b).^2);
    end
end


function constraintHandles = buildEdgeConstraints(terminalCoords, edges, b, p)
% Returns a cell array: constraintHandles{k}(xVec) = ||edge_k||^2 - b^2
    numEdges = size(edges,1);
    constraintHandles = cell(numEdges,1);
    for e = 1:numEdges
        idA = edges(e,1);      % capture IDs for this edge
        idB = edges(e,2);
        constraintHandles{e} = @(x) singleEdgeConstraint( ...
                                   x, idA, idB, terminalCoords, b, p);
    end
end

function g = singleEdgeConstraint(xVec, idA, idB, terminalCoords, b, p)
% evaluates one g(x) = ||edge||^2 - b^2
    S = vecToSteinerMatrix(xVec,p);
    pointA = coordsOfID(idA, terminalCoords, S);
    pointB = coordsOfID(idB, terminalCoords, S);
    g = sum((pointA-pointB).^2) - b^2;
end


%  small utilities --------------------------------------------------

function S = vecToSteinerMatrix(vec, p)
% [S1x;S1y;S2x;S2y] becomes [S1x S1y; S2x S2y]
    if numel(vec) ~= 2*p
        error('Vector length %d should be 2x%d (2P).', numel(vec), p);
    end
    S = reshape(vec, 2, p).';    % Px2
end

function xy = coordsOfID(id, terminalCoords, S)
% returns [x y] for either a terminal (negative ID) or Steiner (positive ID)
    if id < 0
        xy = terminalCoords(-id, :);       % terminal index
    else
        xy = S(id, :);           % steiner index
    end
end


