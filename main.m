clear;
clc;
K = 2;
global train
train = load('train.mat').M1';

% Extract training data
train_samples = train(1:512, 1:3150);
train_labels = train(513, 1:3150);

% Extract testing data
test_samples = train(1:512, 3151:4500);
test_labels = train(513, 3151:4500);

Dte = [test_samples; ones(1, 1350)];

%%%%%%%%%%%%%%%% CG Method %%%%%%%%%%%%%%%%
useCG(train_samples, train_labels, test_labels, Dte);

%%%%%%%%%%%%%%%% BFGS Method %%%%%%%%%%%%%%%%
useBFGS(train_samples, train_labels, test_labels, Dte);

%%%%%%%%%%%%%%%% MLBFGS Method %%%%%%%%%%%%%%%%
useMLBFGS(Dte);

%%%%%%%%%%%%%%%% Newton Method %%%%%%%%%%%%%%%%
useNewton(train_samples, train_labels, test_labels, Dte)

function useCG(train_samples, train_labels, test_labels, Dte)
    train_labels(train_labels == 0)= -1;
    test_labels(test_labels == 0)= -1;

    W0 = zeros(1,513)';
    tic;
    [Ws, f, k] = cg('f_LRBC', 'g_LRBC', W0, train_samples, train_labels, 0.001);
    fprintf("\n\nTraining time for logistic regression cost function with CG (%d iterations): %f seconds\n", k, toc);
    displayResultGeneric(Dte, Ws, test_labels);
end

function useBFGS(train_samples, train_labels, test_labels, Dte)
    train_labels(train_labels == 0)= -1;
    test_labels(test_labels == 0)= -1;

    W0 = zeros(1,513)';
    tic;
    [Ws, f, k] = bfgs('f_LRBC', 'g_LRBC', W0, train_samples, train_labels, 130);
    fprintf("\n\nTraining time for logistic regression cost function with BFGS (%d iterations): %f seconds\n", k, toc);
    
    normw = norm(Ws(1:512));
    for k = 1:513
        Ws(k) = Ws(k)/normw;
    end
    
    displayResultGeneric(Dte, Ws, test_labels);
end

function useMLBFGS(Dte)
    global train
    train_samples = train(1:512, 1:3150);
    train_labels = 1 + train(513, 1:3150);
    Dtr = [train_samples; train_labels];
    test_labels = 1  + train(513, 3151:4500);

    tic;
    [Ws, f]= SRMCC_bfgsML(Dtr, 'f_SRMCC', 'g_SRMCC', 0.002, 2, 130);
    
    normw = norm(Ws(1:512));
    for k = 1:513
        Ws(k) = Ws(k)/normw;
    end
    
    fprintf("\n\nTraining time for softmax regression cost function with ML-BFGS (%d iterations): %f seconds\n",130, toc);
    
    displayResult(Dte, Ws, 2, test_labels);
end

function useNewton(train_samples, train_labels, test_labels, Dte)
    train_labels(train_labels == 0)= -1;
    test_labels(test_labels == 0)= -1;
    
    tic;
    [Ws, f] = LRBC_newton(train_samples, train_labels, 5);
    fprintf("\n\nTraining time for logistic regression cost function with Newton (%d iterations): %f seconds\n", 5, toc);

    displayResultGeneric(Dte, Ws, test_labels);
end

function displayResult(Data_Matrix, Ws, K, ytest)
    [~, ind_pre] = max((Data_Matrix' * Ws)');

    C = zeros(K,K);
    for j = 1:K
        ind_j = find(ytest == j);
        for i = 1:K
            ind_pre_i = find(ind_pre == i);
            C(i,j) = length(intersect(ind_j,ind_pre_i));
        end
    end
    
    disp(C)
    accuracy = sum(diag(C))/(sum(C, 'all')) * 100;
    fprintf("\n\nAccuracy: %f\n", accuracy);
end

function displayResultGeneric(Data_Matrix, Ws, ytest)
    C = zeros(2,2);
    values = sign(Ws' * Data_Matrix);
    
    for value = 1 : length(values)
        if(ytest(value) == 1 && values(value) == 1)
            C(1, 1) = C(1, 1) + 1;
        elseif(ytest(value)== 1 && values(value) == -1)
            C(2, 1) = C(2, 1) + 1;
        elseif(ytest(value) == -1 && values(value) == 1)
            C(1, 2) = C(1, 2) + 1;
        else
            C(2, 2) = C(2, 2) + 1;
        end       
    end

    disp(C)
    accuracy = sum(diag(C))/(sum(C, 'all')) * 100;
    fprintf("\n\nAccuracy: %f\n", accuracy);
end