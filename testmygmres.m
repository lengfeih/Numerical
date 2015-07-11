function testmygmres()
clc
clear
A= [2,3,0,0,0;3,0,4,0,6;0,-1,-3,2,0;0,0,1,0,0;0,4,2,0,1];

for i=1:5
    A(i,i) = A(i,i) + 1e-12;
end

A

b = sum (A,2);
tol = 1e-6;
maxit = 100;


%direct solve
d_x = A \ b;

restart = 2;
x0 = zeros(5,1);

%no preconditioner

precon = eye(5);
[x,err,tot_iter,flag] = myGMRES(A,x0,b,precon,restart,maxit,tol)


%preconditioner is itself
%[x,err,tot_iter,flag] = myGMRES(A,x0,b,A,restart,maxit,tol)


%block diagonal
for i=1:5
    blg(i,i) = A(i,i);
end
[x,err,tot_iter,flag] = myGMRES(A,x0,b,blg,restart,maxit,tol)

%ilu
[L,U] = ilu(sparse(A),struct('type','ilutp','droptol',1e-6));
P = L*U
[x,err,tot_iter,flag] = myGMRES(A,x0,b,P,restart,maxit,tol)


end



function [sol, err, tot_iter, flag] = myGMRES (A, x0, b, M, restrt, max_it, tol)

% GMRES solver without preconditioning

% input   A        REAL nonsymmetric positive definite matrix
%         x0       REAL initial guess vector
%         b        REAL right hand side vector
%         M        REAL preconditioner matrix
%         restrt   INTEGER number of iterations between restart
%         max_it   INTEGER maximum number of iterations
%         tol      REAL converge error tolerance
%
% output  sol      REAL solution vector
%         err      REAL error norm
%         iter     INTEGER number of iterations performed
%         flag     INTEGER: 0 = solution found to tolerance
%                           1 = no convergence given max_it


%Initialization
tot_iter = 0;
flag = 1;


n = length(b);

%with restart, we can save the memory consumption

%L2-orthonormal basis for krylov subspace, use Q in wiki, V = n-by-(restrt+1), the max number of col of V should
%be n
V(1:n, 1:restrt+1) = zeros (n, restrt+1);

%upper Hessenberg matrix, H = (restrt+1)-by-n
H(1:restrt+1,1:restrt) = zeros(restrt+1,restrt);

%???
cs(1:restrt) = zeros(restrt,1);
sn(1:restrt) = zeros(restrt,1);

%
e1 = zeros(n, 1);
e1(1) = 1.0;

sol = x0;

%begin iteration
while (tot_iter < max_it)
    
    r = M \ (b-A*sol);
    V(:,1) = r / norm(r);
    
    beta = norm(r);
    s = beta * e1;
    
    for i = 1:restrt
        tot_iter = tot_iter + 1;
        if(tot_iter >= max_it)
            break;
        end
        %Arnoldi algorithm to get V
        w = M \ (A * V(:,i));
        for k = 1: i
            H(k,i) = w' * V(:,k);
            w = w - H(k,i)*V(:,k);
        end
        H(i+1, i) = norm (w);
        V(:,i+1) = w / H(i+1,i);        
        
        %find yn to minize the norm of residual
        %apply Givens rotation
        for k = 1:i-1                              
            temp     =  cs(k)*H(k,i) + sn(k)*H(k+1,i);
            H(k+1,i) = -sn(k)*H(k,i) + cs(k)*H(k+1,i);
            H(k,i)   = temp;
	    end
        [cs(i),sn(i)] = rotmat( H(i,i), H(i+1,i) ); % form i-th rotation matrix
        temp   = cs(i)*s(i);                        % approximate residual norm
        s(i+1) = -sn(i)*s(i);
        s(i)   = temp;
        H(i,i) = cs(i)*H(i,i) + sn(i)*H(i+1,i);
        H(i+1,i) = 0.0;
        
        err  = abs(s(i+1)) / norm(b);
        if ( err <= tol )                       % update approximation
            y = H(1:i,1:i) \ s(1:i);                 % and exit
            
            sol = sol + V(:,1:i)*y;                 %why acumulate
            flag = 0;
            break;
        end     
    end
    

    if ( flag == 0 )
        break; 
    else %not converge within this restrt iteration
       y = H(1:restrt,1:restrt) \ s(1:restrt);
       sol = sol + V(:,1:restrt)*y;                 
    end

end

end


function [ c, s ] = rotmat( a, b )

%
% Compute the Givens rotation matrix parameters for a and b.
%
   if ( b == 0.0 ),
      c = 1.0;
      s = 0.0;
   elseif ( abs(b) > abs(a) ),
      temp = a / b;
      s = 1.0 / sqrt( 1.0 + temp^2 );
      c = temp * s;
   else
      temp = b / a;
      c = 1.0 / sqrt( 1.0 + temp^2 );
      s = temp * c;
   end

   
end