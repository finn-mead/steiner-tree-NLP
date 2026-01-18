
% buildGvec  -  utility to compute g_i(x) = ||edge_i(x)||^2 - b^2 for all edges
% Inputs:
%   x         : (2Px1) vector of Steiner point coordinates (stacked)
%   termXY    : (Mx2) matrix of terminal coordinates (for negative indices)
%   edgeList  : (Ex2) list of [idA, idB] pairs (negative for terminals)
%   b  : scalar original edge-length bound
%   P  : number of Steiner points
%
% Output:
%   gvec: (Ex1) vector where gvec(e) = ||A - B||^2 - b^2
% -------------------------------------------------------------------------
function gvec = buildGvec(x, termXY, edgeList, b, P)
    gvec = zeros(size(edgeList,1),1);
    S    = reshape(x, 2, P).';  % Px2 matrix of Steiner coords

    for e = 1:size(edgeList,1)
        idA = edgeList(e,1);
        idB = edgeList(e,2);

        if idA < 0
            A = termXY(-idA, :);
        else
            A = S(idA, :);
        end

        if idB < 0
            B = termXY(-idB, :);
        else
            B = S(idB, :);
        end

        gvec(e) = sum((A - B).^2) - b^2;
    end
end
