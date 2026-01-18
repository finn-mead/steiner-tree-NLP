function gradP = gradBarrierObjective(xVec, terminalCoords, edgeList, b, alpha, P)
% gradBarrierObjective  Gradient of barrier-penalized Steiner-tree objective
%
%   xVec           2Px1 vector [s1x; s1y; ...; sPx; sPy]
%   terminalCoords Nx2 matrix of terminal (x,y) coordinates
%   edgeList       Mx2 list of edge endpoints (negative -> terminal, positive -> Steiner)
%   b              maximum edge length
%   alpha          barrier penalty parameter
%   P              number of Steiner points
%
% Returns:
%   gradP          2Px1 gradient of
%                  P_alpha(x) = sum(||edge||^2) - (1/alpha)*sum(log(-g_i(x)))

    % Reshape xVec into Px2 Steiner matrix
    if numel(xVec) ~= 2*P
        error('xVec must have length 2*P.');
    end
    S = reshape(xVec, 2, P).';    % now S(j,:) = [s_jx, s_jy]

    % Initialize accumulators
    grad_f       = zeros(2*P,1);
    grad_penalty = zeros(2*P,1);

    % Loop over each edge
    for e = 1:size(edgeList,1)
        idA = edgeList(e,1);
        idB = edgeList(e,2);

        % --- Fetch coordinates for A ---
        if idA < 0
            A = terminalCoords(-idA, :);  % fixed terminal
        else
            A = S(idA, :);                % Steiner point j
        end

        % --- Fetch coordinates for B ---
        if idB < 0
            B = terminalCoords(-idB, :);
        else
            B = S(idB, :);
        end

        % --- Compute coordinate differences ---
        delX = A(1) - B(1);
        delY = A(2) - B(2);

        % NOTE that in our xVec of size 2p, indexing the partial x of
        % the kth steiner point is given by 2k - 1, and the partial y
        % is indexed by 2k.

        %  Objective gradient f contribution------------------------
        %  For edge (A,B), partials of (A(1)-B(1))^2 + (A(2)-B(2))^2.
        % are 2*delX and 2*delY for point A. for point B, the negative of
        % that.

        if idA > 0
            grad_f(2*idA-1) = grad_f(2*idA-1) + 2*delX;
            grad_f(2*idA  ) = grad_f(2*idA  ) + 2*delY;
        end
        if idB > 0
            grad_f(2*idB-1) = grad_f(2*idB-1) - 2*delX;
            grad_f(2*idB  ) = grad_f(2*idB  ) - 2*delY;
        end

        % Barrier gradient contribution -----------------------------
        g_i = delX^2 + delY^2 - b^2;
        factor = -1 / (alpha * g_i);      
        if idA > 0
            grad_penalty(2*idA-1) = grad_penalty(2*idA-1) + factor*( 2*delX);
            grad_penalty(2*idA  ) = grad_penalty(2*idA  ) + factor*( 2*delY);
        end
        if idB > 0
            grad_penalty(2*idB-1) = grad_penalty(2*idB-1) + factor*(-2*delX);
            grad_penalty(2*idB  ) = grad_penalty(2*idB  ) + factor*(-2*delY);
        end
    end

    % Full barrier-penalized gradient
    gradP = grad_f + grad_penalty;
end