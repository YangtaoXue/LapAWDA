function y=kernel(ker,x1,x2,p)
%Kernel function's computation
%y=kernel(ker,x1,x2,p)
%ker   -------  the type of kernel
%			    	'poly';'rbf';'wave';'scale';'polar','cylinder','sphere'
%x1,x2 ------  the input data;
%p     ------  the parameter of kernel;
%
%Author Zhang Li: zhangli@rsp.xidian.edu.cn
global p1 p2

if nargin<4   p=p1;  end;

l=size(x2);
l1=size(x1);
l1=l1(1);
e=ones(l1,1);
y=zeros(l1,l(1));
switch lower(ker)
case 'linear'
	y=x1*x2'; 
case 'poly'
   y=(x1*x2').^p;
case 'sigmoid'
    y = tanh(p1.*(x1*x2') + p2);
case 'rbf'
    [L1, dim] = size(x1);
    [L2, dim] = size(x2);
    a = sum(x1.*x1,2);
    b = sum(x2.*x2,2);
    dist2 = a*ones(1,L2) + ones(L1,1)*b' - 2*x1*x2';
    y= exp(-p.*dist2);

% %   if length(p)==1
%      if any(imag(x1)~=0)
%       for k=1:l1
%          for j=1:l(1)
%             %y(k,j)=exp(-((x1(k,:)-x2(j,:))*(x1(k,:)-x2(j,:))')./(2*p^2));
%             y(k,j)=exp(-((x1(k,:)-x2(j,:))*(x1(k,:)-x2(j,:))').*p);
%          end
%       end
%    elseif l(2)==1
%       for k=1:l(1)
%          temp=(x1-x2(k));
%          y(:,k)=exp(-(temp.*temp).*p);
%       end
%      else
%         if length(p)==1
%             p=ones(1,size(x1,2))*p;
%         end
%       for k=1:l(1)
%          temp=(x1-e*x2(k,:));
%          y(:,k)=exp(-(p*(temp.*temp)'));       
%       end
%      end  
%   else
%       if any(imag(x1)~=0)
%       for k=1:l1
%          for j=1:l(1)
%             %y(k,j)=exp(-((x1(k,:)-x2(j,:))*(x1(k,:)-x2(j,:))')./(2*p^2));
%             y(k,j)=exp(-(p.*(x1(k,:)-x2(j,:))*(x1(k,:)-x2(j,:))'));
%          end
%       end
%    elseif l(2)==1
%       for k=1:l(1)
%          temp=(x1-x2(k));
%          y(:,k)=exp(-(p.*temp.*temp));
%       end
%    else
%       for k=1:l(1)
%          temp=(x1-e*x2(k,:));
%          y(:,k)=exp(-sum((temp.*temp)').*p)';       
%       end
%    end  
%   end
case 'wave'
   if length(p)==1
      p=ones(l(1),1)*p;
   end
   if l(2)==1
      for k=1:l(1)
         temp=(x1-x2(k))/p(k);
         y(:,k)=cos(1.75.*((temp))).*exp(-0.5.*(temp.*temp));
      end
   else
      for k=1:l1(1)
         for j=1:l(1)
            temp=(x1(k,:)-x2(j,:))/p(j);
            y(k,j)=prod(cos(1.75.*(temp)).*exp(-0.5.*(temp.*temp)));
         end         
      end      
   end
case 'scale'
   for i=1:l1
      for j=1:l(1)
         y(i,j)=prod(sinc(p*(x1(i,:)-x2(j,:))));
      end
   end   
case {'polar','cylinder','sphere'}
   l1=size(x1,1);
   l2=size(x2,1);
   for i=1:l1
      for j=1:l2
         pp1=norm(x1(i,:));
         pp2=norm(x2(j,:));
         if x1(i,1)==0
            sita1=pi/2;
         else
            sita1=atan(x1(i,2)/x1(i,1));
         end
         if x2(j,1)==0
            sita2=pi/2;
         else
            sita2=atan(x2(j,2)/x2(j,1));
         end
         if (x1(i,1)<0)
            sita1=sita1+pi;
         elseif (x1(i,2)<0)&(x1(i,1)>0)
            sita1=sita1+2*pi;
         end
         if (x2(j,1)<0)
            sita2=sita2+pi;
         elseif (x2(j,2)<0)&(x2(j,1)>0)
            sita2=sita2+2*pi;         
         end   
         switch lower(ker)
         case 'polar'
            y(i,j)=pp1*pp2+sita1*sita2;
         case 'cylinder'
            y(i,j)=x1(i,3)*x2(j,3)+sita1*sita2+norm(x1(i,1:2))*norm(x2(j,1:2));
         case 'sphere'
            phi1=acos(x1(i,3)/pp1);
            phi2=acos(x2(j,3)/pp2);
            y(i,j)=pp1*pp2+phi1*phi2+sita1*sita2;
         end
      end
   end
end

