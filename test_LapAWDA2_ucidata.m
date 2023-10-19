function test_LapAWDA2_ucidata(dataset)

% dataset = 'balance';
% kfanwei = 5;
kfanwei = [1 3 5 7];
fanwei = (-5:1:5);
n_t = 30;

for loop = 1:10
    disp(['The loop of ' num2str(loop) '...']);
    %%% fix data
    p_l = 0.2; p_ul = 0.5;
    
    disp('The grid search is in progress ...');
    
%     load(['./ucidata73/' dataset '_' num2str(loop) '.mat']);
    load(['./multi_ucidata73/' dataset '_' num2str(loop) '.mat']);
    c = unique(Y_trn);
    m = size(X_trn,2);
    if m < length(c)
        dim = m-1;
    else
        dim = length(c)-1;
    end
%   dim = length(c)-1;
    
%     p1 = kernelpar(X_trn);
    x_trn = []; y_trn = []; x_trn_l = []; y_trn_l = [];
    for i = 1:length(c)
        in = find(Y_trn == c(i));
        n0 = length(in);
        n_l = ceil(p_l*n0); n_ul = ceil(p_ul*n0)-1;
        x_trn = [x_trn; X_trn(in(1:n_l+n_ul),:)];
        y_trn = [y_trn; c(i)*ones(n_l,1); zeros(n_ul,1)];
        x_trn_l = [x_trn_l; X_trn(in(1:n_l),:)];
        y_trn_l = [y_trn_l; c(i)*ones(n_l,1)];
    end
    
    for k = 1:length(kfanwei)
        Knn = kfanwei(k);
        for c1 = 1:length(fanwei)
            alpha = 2^fanwei(c1);
            for c2 = 1:length(fanwei)
                beta = 2^fanwei(c2);
            [W, ~] = LapAWDA2(x_trn, y_trn, dim, n_t, Knn, alpha, beta);
            [~, acc(k,c1,c2)] = knnclassifier(x_trn_l*W, y_trn_l, X_tst*W, Y_tst, 1);
            end
        end
    end
    
    [a_m, in] = max(acc(:));
    [in1 in2 in3] = ind2sub(size(acc), find(acc >= a_m));
    
    disp('The test is in progress');
    for j = 1: length(in1)
        Knn = []; alpha = []; beta = []; W = [];
        Knn = kfanwei(in1(j));
        alpha = 2^fanwei(in2(j));
        beta = 2^fanwei(in3(j));
        t0 = cputime;
        [W, ~] = LapAWDA2(x_trn, y_trn, dim, n_t, Knn, alpha, beta);
        [~, ACC(j)] = knnclassifier(x_trn_l*W, y_trn_l, X_tst*W, Y_tst, 1);
        tt(j) = cputime-t0;
    end
    [test_acc(loop), in00(loop)] = max(ACC);  test_t(loop) = tt(in00(loop));
end
testacc = mean(test_acc)
teststd = std(test_acc)
test_tt = mean(test_t)


save(['./results_ucidata73/lapawda_non_' dataset '_' num2str(p_l*10) num2str(p_ul*10) '.mat']);
clear;
