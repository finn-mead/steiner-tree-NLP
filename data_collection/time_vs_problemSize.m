function time_vs_problemSize()
    % time_vs_problemSize  Compare BFGS (P1+P2) vs fmincon (P2 from zeros)
    % across varying problem sizes, recording runtimes and failure rates.
    rng(3);

    % problem sizes
    sizesBoth      = [10, 25, 50, 100, 250, 500, 1000];
    sizesBFGSOnly  = [];
    allSizes       = unique([sizesBoth, sizesBFGSOnly]);
    nBoth          = numel(sizesBoth);
    nAll           = numel(allSizes);

    Nwant          = 20;      % instances per size
    tolF           = 1e-8;    % feasibility tolerance for checks

    % preallocate
    tBFGS = nan(nAll, Nwant);
    tFMC  = nan(nBoth, Nwant);

    runsBFGS = zeros(nAll,1);
    failBFGS = zeros(nAll,1);
    runsFMC  = zeros(nBoth,1);
    failFMC  = zeros(nBoth,1);

    fprintf('=== Timing vs. Problem Size ===\n');

    % 1) BFGS pipeline (P1+P2) on allSizes
    for ii=1:nAll
        nT = allSizes(ii);
        P  = max(1,nT-2);

        fprintf('BFGS pipeline: nT=%d\n',nT);
        for inst=1:Nwant
            fprintf("%d/%d for size %d\n", inst, Nwant, ii);
            runsBFGS(ii)=runsBFGS(ii)+1;
            % generate
            [T,~,E,x0,b0] = generateSteinerInstance(nT,100, [0.6, 0.7]);
            % P1
            t1 = tic;
            [xFeas,~,okP1] = findFeasibleStart(T,E,b0,P,x0,tolF);
            t1 = toc(t1);
            % P2
            t2 = Inf; okP2=false;
            if okP1 && (~isempty(xFeas)||P==0)
                t2 = tic;
                [xOpt,fB,~] = solveSteinerBarrierBFGS(T,E,b0,P,xFeas);
                okP2 = all(buildGvec(xOpt,T,E,b0,P)<-tolF);
                t2 = toc(t2);
            end
            if okP1 && okP2
                tBFGS(ii,inst)= t1 + t2;
            else
                failBFGS(ii)=failBFGS(ii)+1;
            end
        end
    end

    % 2) fmincon P2 (from zeros) on sizesBoth
    for jj=1:nBoth
        nT = sizesBoth(jj);
        P  = max(1,nT-2);

        fprintf('fmincon P2: nT=%d\n',nT);
        for inst=1:Nwant
            runsFMC(jj)=runsFMC(jj)+1;
            % generate
            [T,~,E,~,b0] = generateSteinerInstance(nT,100,[0.8,1.5]);
            % P2 from zeros
            t0 = tic;
            [obj, cn] = steinerNLP(T,E,b0,P);
            nl = @(x) edgeNonlcon(x,cn);
            opts = optimoptions('fmincon','Display','off','ConstraintTolerance',1e-7);
            try
                [xF,fF,ef] = fmincon(obj,zeros(2*P,1),[],[],[],[],[],[],nl,opts);
                gF = buildGvec(xF,T,E,b0,P);
                okF = ef>=0 && all(gF<-tolF);
            catch
                okF = false;
                fF = NaN;
            end
            tf = toc(t0);
            if okF
                tFMC(jj,inst) = tf;
            else
                failFMC(jj)=failFMC(jj)+1;
            end
        end
    end

    % 3) compute averages
    avgBFGS = mean(tBFGS,2,'omitnan');
    avgFMC  = mean(tFMC,2,'omitnan');

    % 4) plot
    figure; hold on;
    plot(allSizes, avgBFGS, '-o','LineWidth',1.5,'DisplayName','BFGS P1+P2');
    plot(sizesBoth, avgFMC, '-s','LineWidth',1.5,'DisplayName','fmincon P2 from 0');
    set(gca,'YScale','log');
    xlabel('Number of Terminals, N_T');
    ylabel('Avg. Time (s)');
    title(sprintf('Solver Time vs. Problem Size (N= %d)',Nwant));
    legend('Location','NorthWest');
    grid on; hold off;

    % 5) print failure tables
    fprintf('\nBFGS pipeline failures:\n');
    fprintf('  nT | runs | fails | fail rate (%%)\n');
    for ii=1:nAll
        rate = 100*failBFGS(ii)/runsBFGS(ii);
        fprintf('%4d | %4d | %5d | %9.2f\n', allSizes(ii), runsBFGS(ii), failBFGS(ii), rate);
    end

    fprintf('\nfmincon P2 failures:\n');
    fprintf('  nT | runs | fails | fail rate (%%)\n');
    for jj=1:nBoth
        rate = 100*failFMC(jj)/runsFMC(jj);
        fprintf('%4d | %4d | %5d | %9.2f\n', sizesBoth(jj), runsFMC(jj), failFMC(jj), rate);
    end
end