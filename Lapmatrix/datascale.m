function [X_trn, Y_trn, X_tst, Y_tst] = datascale(x_trn,y_trn,x_tst,y_tst)

c = unique(y_trn);
X_trn = []; Y_trn = []; X_tst = []; Y_tst = [];
for i = 1:length(c)
    in1 = []; x = []; y = []; mu = []; sigma =[];
    in1 = find(y_trn == c(i));
    [x, mu, sigma] = zscore(x_trn(in1,:));
    X_trn = [X_trn; x];
    y = c(1)*ones(length(in1),1);
    Y_trn = [Y_trn; y];
    if isempty(x_tst)
        X_tst = []; Y_tst = [];
    else
        in2 = find(y_tst == c(i));
        x_temp = (x_tst(in2,:)-repmat(mu,length(in2),1))./repmat(sigma,length(in2),1);
        y_temp = c(1)*ones(length(in2),1);
        X_tst = [X_tst; x_temp];
        Y_tst = [Y_tst; y_temp];
    end  
end

    