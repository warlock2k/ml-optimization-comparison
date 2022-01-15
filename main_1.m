clc;
clear;

samples_train = 16000;
samples_test = 10000;

% Representing number of classes
K = 10;

Raw_Training_Data = load('X1600.mat').X1600;
Raw_Testing_Data = load('Te28.mat').Te28;
Raw_Testing_Data_Labels = load('Lte28.mat').Lte28;
u = ones(1, samples_train/10);

% Building y training dataset such that
% 1- 1600 belong to class 1
% 1601 - 3200 belong to class 2... and so on ...
Raw_Training_Data_Labels = [u 2*u 3*u 4*u 5*u 6*u 7*u 8*u 9*u 10*u];
Dtr = [Raw_Training_Data(:, 1: samples_train); Raw_Training_Data_Labels(1: samples_train)];
Dte = [Raw_Testing_Data(:, 1: samples_test); 1 + Raw_Testing_Data_Labels(1: samples_test)'];

% 5.2 Using Dtr, Dte, mu = 0.002 & iter = 62
[Ws, f]= SRMCC_bfgsML(Dtr, 'f_SRMCC', 'g_SRMCC', 0.002, K, 10);
displayResult(Dte, Ws, K);

H = [];
fh = @hog20;
memoizedFcn = memoize(fh);

for i = 1: samples_train
    image_vector = Raw_Training_Data(:, i);
    img = reshape(image_vector, 28, 28);
    hog_descriptior_image = memoizedFcn(img, 7, 9);
    H = [H hog_descriptior_image];
end

% Data Matrices DHtr representing training data and its corresponding
% labels.
DHtr = [H; Raw_Training_Data_Labels(1:samples_train)];
Hte = [];

for i = 1:samples_test
    img_vector = Raw_Testing_Data(:,i);
    img = reshape(img_vector, 28, 28);
    hog_descriptor = memoizedFcn(img, 7, 9);
    Hte = [Hte hog_descriptor];
end

Dhte = [Hte; 1+Raw_Testing_Data_Labels(1:samples_test)'];

% 5.4 Using DHtr, DHte, mu = 0.002 & iter = 62
[Ws, f]= SRMCC_bfgsML(DHtr, 'f_SRMCC', 'g_SRMCC', 0.001, K, 57);
displayResult(Dhte, Ws, K)



function displayResult(Data_Matrix, Ws, K)
    tic;
    [~,ind_pre] = max((Data_Matrix' * Ws)');
    ytest = Data_Matrix(size(Data_Matrix, 1), :);

    C = zeros(K,K);
    for j = 1:K
        ind_j = find(ytest == j);
        for i = 1:K
            ind_pre_i = find(ind_pre == i);
            C(i,j) = length(intersect(ind_j,ind_pre_i));
        end
    end
    fprintf('\nTime taken for classification of %d samples is %f seconds\n\n', length(ytest), toc);
    
    disp(C)
    accuracy = sum(diag(C))/(sum(C, 'all')) * 100;
    fprintf("\n\nAccuracy: %f\n", accuracy);
end
