% setupPhase1NLP_fmincon  -  sets up phase 1 problem with explicit
% constraints as opposed to barrier method. used for fmincon.
% 
% Decision vector: y = [xSteiner (2Px1); s] 
% Problem: minimize s
%          subject to g_i(x) - s <= 0  for all original constraints g_i(x).
% -------------------------------------------------------------------------
function [phase1_obj_for_fmincon, phase1_nonlcon_for_fmincon] = setupPhase1NLP_fmincon(...
    terminalCoords, edgeList, bOriginal, numSteinerPoints)

    % 1. Get handles to original constraint functions g_i(x)
    [~, original_g_handles] = steinerNLP(terminalCoords, edgeList, bOriginal, numSteinerPoints);

    % 2. Objective handle for fmincon: minimize s (last element of y)
    phase1_obj_for_fmincon    = @objective_phase1_constrained;

    % 3. Nonlinear constraints: g_i(x) - s <= 0
    phase1_nonlcon_for_fmincon = @constraints_phase1_constrained;

    % ---------------------------------------------------------------------
    % Phase-1 objective (for fmincon)
    function s_value = objective_phase1_constrained(y_phase1_vec)
        % y_phase1_vec = [x_steiner_coords; s]
        s_value = y_phase1_vec(end);
    end

    % ---------------------------------------------------------------------
    % Phase-1 nonlinear constraints (for fmincon)
    function [c_ineq_phase1, ceq_eq_phase1] = constraints_phase1_constrained(y_phase1_vec)
        % y_phase1_vec = [x_steiner_coords; s]
        if numSteinerPoints > 0
            x_steiner_part = y_phase1_vec(1 : 2*numSteinerPoints);
        else
            x_steiner_part = [];
        end
        s_auxiliary = y_phase1_vec(end);

        nCon = numel(original_g_handles);
        c_ineq_phase1 = zeros(nCon,1);
        for k = 1:nCon
            c_ineq_phase1(k) = original_g_handles{k}(x_steiner_part) - s_auxiliary;
        end
        ceq_eq_phase1 = [];
    end
end
