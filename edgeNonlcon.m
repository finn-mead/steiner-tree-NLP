% tiny helper for parsing my constraint functions convention into
% one compatible with built in matlab solver fmincon

function [c, ceq] = edgeNonlcon(x, constraintFuns)
    % constraintFuns is a cell array {g1(x), g2(x), ...}
    M = numel(constraintFuns);
    c = zeros(M,1);
    for i = 1:M
        c(i) = constraintFuns{i}(x);
    end
    ceq = [];   % no equality constraints
end