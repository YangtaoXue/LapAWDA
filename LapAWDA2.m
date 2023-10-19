function [W, alpha] = LapAWDA2(X_trn, Y_trn, dim, n_t, Knn, alpha, beta)
%%% add a manifold regularization and L2,1 norm on projection matrix
[n,m] = size(X_trn);
c = unique(Y_trn(find(Y_trn~=0)));
n_c = length(c);
X_trn_l = X_trn(find(Y_trn~=0),:);
n_l = size(X_trn_l,1);
u = mean(X_trn_l,1);
% alpha = 2/(n_c*(n_c-1));
% u = mean(X_trn,1);
Sb = zeros(m); Sw = zeros(m);

options=ml_options('NN',Knn,'GraphWeights','binary');
% p = kernelpar(X_trn);
% options=ml_options('NN',Knn,'GraphWeights','heat', 'GraphWeightParam',p);
options.GraphNormalize = 0;
Lt = laplacian(X_trn,'nn',options);
% A = agraph(X_trn',Y_trn,Knn);
% D = sum(A(:,:),2);
% Lt = diag(D) - A;
E = beta*X_trn'*Lt*X_trn;

for i = 1:n_c
    in0 = find(Y_trn == c(i));
    Data{1,i}.X_trn = X_trn(in0,:);
    Data{1,i}.n = length(in0);
    Data{1,i}.u = mean(Data{1,i}.X_trn,1);
    Data{1,i}.Sw =Data{1,i}.n*(Data{1,i}.X_trn - repmat(Data{1,i}.u,Data{1,i}.n,1))' * (Data{1,i}.X_trn -repmat(Data{1,i}.u,Data{1,i}.n,1));
    Data{1,i}.Sb = Data{1,i}.n * (u- Data{1,i}.u)' * (u - Data{1,i}.u);
    Sb = Sb + Data{1,i}.Sb;
    Sw = Sw + Data{1,i}.Sw;
end
Sb = Sb./n; Sw = 1/n_l*(Sw+1e-6*eye(m));
in_cc = nchoosek(1:n_c,2);
n_cc = size(in_cc,1);
M = zeros(m);

for j = 1:n_cc
    in1 = in_cc(j,1); in2 = in_cc(j,2);
    S{1,j}.B = (Data{1,in1}.u-Data{1,in2}.u)'*(Data{1,in1}.u-Data{1,in2}.u);
    S{1,j}.a = 2/(n_c*(n_c-1));
    a(j) = 2/(n_c*(n_c-1));
    S{1,j}.Q = S{1,j}.B;  %%S{1,j}.Q = inv(Sw)*S{1,j}.B;
    S{1,j}.M = S{1,j}.Q./a(j);
    M = M+S{1,j}.M;
end

T{1,1}.M = (M'+M)./2; T{1,1}.a = a; T{1,1}.F = 0;
%%%
% [V1,D1] = eig(inv(Sw)*Sb)
% [~,in1] = sort(diag(D1),'descend');
% T{1,1}.V = V1(:,in1(1:dim));
T{1,1}.V = 0;

for t = 2:n_t
    T{1,t}.V = UpdateV(S, T{1,t-1}.a, E,T{1,t-1}.V,alpha,m,n_cc,dim);
    T{1,t}.a = UpdateA((Sw^(-1/2))*T{1,t}.V, S, n_cc);
    a_s = sum(T{1,t}.a);
%     tempppp = norm(T{1,t}.V-T{1,t-1}.V)
    if t == 1
        continue;
    else if (norm(T{1,t}.V-T{1,t-1}.V) <= 1e-3)
            break;
        end
    end
end
W = T{1,t}.V; alpha = T{1,t-1}.a;

function V = UpdateV(S, a, E,W, alpha,m,n,dim)
d = sqrt(sum(W.*W,2)+eps);
D = diag(0.5/d);
M = zeros(m);
for i = 1:n
    M0 = S{1,i}.Q./a(i);
    M = M+M0;
end
M = (M'+M)./2;
H = M-E-alpha*D;
[V,D00] = eigs(H,dim);

% [V0,D0] = eig(H);
% [~,in1] = sort(diag(D0),'descend');
% V = V0(:,in1(1:dim));



function [A,a_s] = UpdateA(V, S, n_cc)
   for j = 1:n_cc
        a(j) = sqrt(trace(V'*S{1,j}.Q*V));
        a(j) = max(a(j),1e-6);
    end
    a_s = sum(a);
%     a_s = 1;
    for j = 1:n_cc
        A(j) = a(j)./a_s;
%         A(j) = (2/(n_cc*(n_cc-1)))*a(j)^2/a_s^2;
    end

