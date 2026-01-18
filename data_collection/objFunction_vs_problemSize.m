function objFunction_vs_problemSize()
    thisDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(thisDir,'..'));

    rng(1);
    sizes        = [10, 20, 30, 40, 50];
    nInstances   = 100;
    nSizes       = numel(sizes);

    data.sizes              = sizes;
    data.bFGS               = nan(nSizes, nInstances);
    data.fmc                = nan(nSizes, nInstances);
    infeasCount             = zeros(nSizes,1);
    falseNegCount           = zeros(nSizes,1);
    falsePosCount           = zeros(nSizes,1);
    bfgsOutputInfeasCount   = zeros(nSizes,1);

    for i = 1:nSizes
        nTerm = sizes(i);
        P     = nTerm - 2;
        fprintf("Running size %d (%d instances)...\n", nTerm, nInstances);
        for j = 1:nInstances
            fprintf("Size %d, instance %d/%d\n", nTerm, j, nInstances);
            [termXY,nodeIDs,edgeList,xGuess,bVal] = generateSteinerInstance(nTerm,100, [0.5, 0.6]);

            % Phase-1 evaluation
            [okB1, okF1, xFeasB, ~] = phase1Evaluate(termXY, edgeList, bVal, P, xGuess);

            % Phase-1 stats
            if     okB1 && okF1,  infeasCount(i)=infeasCount(i);
            elseif ~okB1 && ~okF1, infeasCount(i)=infeasCount(i)+1;
            elseif ~okB1 && okF1,  falseNegCount(i)=falseNegCount(i)+1;
            elseif okB1 && ~okF1, falsePosCount(i)=falsePosCount(i)+1;
            end

            % Phase-2 only if Phase-1 BFGS succeeded
            if okB1
                [fB2, okB2] = phase2BFGS(termXY,edgeList,bVal,P,xFeasB);
                [fF2, okF2] = phase2Fmincon(termXY,edgeList,bVal,P,xFeasB);
                data.bFGS(i,j) = fB2;
                data.fmc(i,j)  = fF2;
                if ~okB2 && ~isnan(fB2)
                    bfgsOutputInfeasCount(i) = bfgsOutputInfeasCount(i) + 1;
                end
            else
                data.bFGS(i,j) = NaN;
                data.fmc(i,j)  = NaN;
            end
        end
    end

    % 1. Compute per‐instance relative error (in %)
    relErrInst = 100*(data.bFGS - data.fmc) ./ data.fmc;

    % 2. Average across instances (omit NaNs for infeasible cases)
    meanRelErr = mean(relErrInst, 2, 'omitnan');

    pctInfeasPhase1    = 100 * infeasCount           / nInstances;
    pctFalseNegPhase1  = 100 * falseNegCount         / nInstances;
    pctFalsePosPhase1  = 100 * falseNegCount         / nInstances;
    pctInfeasPhase2Out = 100 * bfgsOutputInfeasCount / nInstances;

    % Plot objective gap vs. size
    % 3. Plot it
    figure;
    plot(sizes, meanRelErr, '-^', 'LineWidth', 1.5);
    yline(0, '--k', 'LineWidth', 1);
    xlabel('Number of Terminals');
    ylabel('Avg. % Error per Instance');
    title('Mean Instance‐wise % Error vs. Problem Size');
    grid on;

    % Display feasibility statistics
    fprintf('\nPhase-1 feasibility stats by problem size:\n');
    fprintf(' nTerm | infeas%% | falseNeg%%\n');
    for i = 1:nSizes
        fprintf('  %3d  |   %6.2f   |   %6.2f\n', sizes(i), pctInfeasPhase1(i), pctFalseNegPhase1(i));
    end
    fprintf('\nPhase-2 BFGS output infeasible rates:\n');
    fprintf(' nTerm | infeasOut%%\n');
    for i = 1:nSizes
        fprintf('  %3d  |   %6.2f\n', sizes(i), pctInfeasPhase2Out(i));
    end

end

function [okB, okF, xFeas, sOptB] = phase1Evaluate(termXY,edgeList,bVal,P,xGuess)
    epsFeas = 1e-6;
    % BFGS Phase-1
    [xFeas, sOptB, okB] = findFeasibleStart(termXY,edgeList,bVal,P,xGuess,epsFeas);
    % fmincon Phase-1
    [objF1, conF1] = setupPhase1NLP_fmincon(termXY,edgeList,bVal,P);
    y01 = [xGuess; max(0,max(buildGvec(xGuess,termXY,edgeList,bVal,P))) + 1e-3];
    opts1 = optimoptions('fmincon','Algorithm','interior-point','Display','off', ...
                        'ConstraintTolerance',1e-7,'StepTolerance',1e-7);
    try
        [yOpt,~,exitflag] = fmincon(objF1,y01,[],[],[],[],[],[],conF1,opts1);
    catch
        exitflag = -1; yOpt = y01;
    end
    sOptF = yOpt(end);
    gF1   = buildGvec(yOpt(1:end-1),termXY,edgeList,bVal,P);
    okF   = (sOptF < -epsFeas) && all(gF1 < -epsFeas);
    if ~okF && exitflag>=0 && (sOptF < -epsFeas) && max(gF1) < epsFeas
        okF = true;
    end
end

function [fVal, okPhase2] = phase2BFGS(termXY,edgeList,bVal,P,xFeas)
    [xOpt, fVal, ~] = solveSteinerBarrierBFGS(termXY,edgeList,bVal,P,xFeas);
    gVals = buildGvec(xOpt,termXY,edgeList,bVal,P);
    okPhase2 = all(gVals < 0);
    if ~okPhase2, fVal = NaN; end
end

function [fVal, okPhase2] = phase2Fmincon(termXY,edgeList,bVal,P,xFeas)
    [obj, cn] = steinerNLP(termXY,edgeList,bVal,P);
    nonl = @(x) edgeNonlcon(x,cn);
    opts2 = optimoptions('fmincon','Algorithm','interior-point','Display','off', ...
                        'ConstraintTolerance',1e-7,'StepTolerance',1e-7);
    try
        [~, fVal, exitflag2] = fmincon(obj,xFeas,[],[],[],[],[],[],nonl,opts2);
        okPhase2 = exitflag2 ~= -2;
    catch
        fVal = NaN; okPhase2 = false;
    end
end