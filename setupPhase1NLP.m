% setupPhase1NLP  -  Phase-1 log-barrier objective + explicit-loop gradient
% ---------------------------------
% Decision vector
%   y = [xSteiner (2Px1); s] .
%
% Objective
%   P_alpha(y) = s  - (1/alpha) * Sum over i (log( s - g_i(x) ))
%               - (1/alpha) * log( s_upper_bound - s )
%
% Gradient built by looping over edges
%   Grad_x P_alpha = (1/alpha) * Σ_i  [ Grad_x(g_i(x)) / (s - g_i(x)) ]
%   Final component, d/ds P_alpha = 1 - (1/alpha) * Σ_i 1/(s - g_i(x))
%                              + (1/alpha) * 1/(s_upper_bound - s)
% 
function [objFun, gradFun, gVecFun] = ...
    setupPhase1NLP(terminalCoords, edgeList, bOriginal, numSteinerPoints, alpha, s_upper_bound)

if nargin < 5, alpha = 1.0; end
if nargin < 6
    error('s_upper_bound must be provided to setupPhase1NLP for the s-cap.');
end

% Handle to vector g(x) (external utility)
gVecFun = @(x) buildGvec(x, terminalCoords, edgeList, bOriginal, numSteinerPoints);

% Objective handle --------------------------------------------------------
objFun  = @(y) phase1Obj(y, alpha, gVecFun, s_upper_bound);

% Gradient handle (loop over edges) --------------------------
gradFun = @(y) phase1Grad(y, alpha, terminalCoords, edgeList, ...
                         bOriginal, numSteinerPoints, s_upper_bound);
end


%  Local helpers
% =======================================================================
function Pval = phase1Obj(y, alpha, gHandle, s_upper_bound)
    x = y(1:end-1);     s = y(end);
    delta = s - gHandle(x);              % s - g_i(x)  (vector, >0)
    cap   = s_upper_bound - s;           % s_upper_bound - s  (scalar)

    if any(delta <= 1e-10) || cap <= 1e-10
        Pval = inf;
        return;
    end

    Pval = s ...
         - (1/alpha) * sum(log(delta)) ...
         - (1/alpha) * log(cap);
end

% essentially the same idea as gradBarrierObjective.m
function gradY = phase1Grad(y, alpha, termXY, edgeList, b, P, s_upper_bound)
    x = y(1:end-1);     s = y(end);
    gradX     = zeros(2*P,1);           % accumulate Gradient_x part
    sumInv    = 0;                       % accumulate Σ_i 1/(s-g_i)

    S = reshape(x,2,P).';               % Px2 matrix
    for e = 1:size(edgeList,1)
        idA = edgeList(e,1);  idB = edgeList(e,2);

        if idA < 0, A = termXY(-idA,:); else, A = S(idA,:); end
        if idB < 0, B = termXY(-idB,:); else, B = S(idB,:); end

        dX  = A(1) - B(1);    dY  = A(2) - B(2);
        g_i = dX^2 + dY^2 - b^2;
        delta = s - g_i;                  % strictly positive

        if delta <= 1e-10
            invDelta = 1/(sign(delta+eps)*eps + delta);
        else
            invDelta = 1/delta;
        end
        sumInv = sumInv + invDelta;
        coeff  = (1/alpha) * invDelta;

        if idA > 0
            gradX(2*idA-1) = gradX(2*idA-1) + coeff * (2*dX);
            gradX(2*idA  ) = gradX(2*idA  ) + coeff * (2*dY);
        end
        if idB > 0
            gradX(2*idB-1) = gradX(2*idB-1) + coeff * (-2*dX);
            gradX(2*idB  ) = gradX(2*idB  ) + coeff * (-2*dY);
        end
    end

    % s-gradient from original barrier term
    gradS = 1 - (1/alpha) * sumInv;
    % add gradient from upper-bound barrier
    cap = s_upper_bound - s;
    if cap <= 1e-10
        invCap = 1/(sign(cap+eps)*eps + cap);
    else
        invCap = 1/cap;
    end
    gradS = gradS + (1/alpha) * invCap;

    gradY = [gradX ; gradS];
end
