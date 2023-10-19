function [par]=kernelpar(X,method,Y)
%
% 'meidian' ????????: L. Zhang*, W.-D. Zhou, P.-C. Chang, J. Liu, Z. Yan,
% T. Wang and F.-Z. Li.  Kernel sparse representation-based classifier, 
% IEEE Transactions on Signal Processing, 2012, 60, 1684-1695.
%
%'new'????????: Z. Xu, M. Dai, D. Meng, Fast and efficient strategies for model
%selection of Gaussian support vector machine, IEEE Transactions on System,
%man, and cybernetics-Part B: cybernetics, 2009, 39(5): 1292-1307.
%
if nargin<3 Y=0;end
if nargin<2 method='median';end
switch lower(method)
    case 'median'
        [L,D] = size(X);
        x=mean(X);b=[];
        for i=1:L
            a=norm(X(i,:)-x);
            b=[b,a];
        end
        mx=median(b);
        par =1/(mx^2);
    case 'xunew'
        L=size(X,1);
        a=0.1;
        b=1;
        index=randperm(L);
        X1=X(index(1:ceil(b*L)),:);
        D=pdist2(X1,X,'euclidean');
        [D,index]=sort(D(:));
        up=floor((1-a)*b*L*(L-1));
        down=floor(a*b*L*(L-1)+1);
        d1=D(up);
        d2=D(down);
        par=(log(d1^2)-log(d2^2))/(d1^2-d2^2);
    case 'mynew1'
        L=size(X,1);
        a=0.25; b=1;
        index=randperm(L);
        X1=X(index(1:ceil(b*L)),:);
        D=pdist2(X1,X,'euclidean');
        [D,index]=sort(D(:));
        D(D==0)=[];
        up=floor((1-a)*length(D));
        down=floor(a*length(D));
        d1=D(up);
        d2=D(down);
        par=(log(d1^2)-log(d2^2))/(d1^2-d2^2);
        
    case 'mynew'
        D=pdist2(X,X,'euclidean');
        classnum=unique(Y);
        DMin=[];
        DMax=[];
        for i=1:length(classnum)
            in2=[1:length(Y)];
            in1=find(Y==classnum(i));
            in2(in1)=[];
            [D1,index1]=sort(D(in1,in1));
            DMin=[DMin D1(end,:)];
            [D2,index2]=sort(D(in1,in2));
            DMax=[DMax D2(2,:)];
        end
        DMax=mean(DMax.^2);DMin=mean(DMin.^2);
        par=(log(DMin)-log(DMax))/(DMin-DMax);
end

function [f,g] = myfun(x,DMax,DMin)
f = -mean(exp(-DMin./(x^2))-exp(-DMax./(x^2))); % function
g = -mean(exp(-DMin./(x^2)).*(DMin./(x^3))-exp(-DMax./(x^2)).*(DMax./(x^3)));