% Generate Following Tridiagonal Matrix 
% Descriptions are copy of Table 1. by Christoph Klein and Robert Strzodka. 2021. 
% Tridiagonal GPU Solver with Scaled Partial Pivoting at Maximum Bandwidth. 
% In 50th International Conference on Parallel Processing (ICPP 2021). 
% Association for Computing Machinery, New York, NY, USA, Article 54, 1–10. 
% DOI:https://doi.org/10.1145/3472456.3472484
% 
% 1. tridiag(a, b, c) with a, b, c sampled from U (−1, 1)
% 2. b=1e+8*ones(N, 1); a, c sampled from U (−1, 1)
% 3. gallery(’lesp’, N)
% 4. same as #1, but a(N/2+1, N/2) = 1e-50*a(N/2+1,N/2)
% 5. same as #1, but each element of a, c has 50% chance to be zero 
% 6. 64*ones(N,1); a, c sampled from U (−1, 1)
% 7. inv(gallery(’kms’, N, 0.5)) Toeplitz, inverse of Kac-Murdock-Szegö 
% 8. gallery(’randsvd’, N, 1e15, 2, 1, 1)
% 9. gallery(’randsvd’, N, 1e15, 3, 1, 1) 
% 10. gallery(’randsvd’, N, 1e15, 1, 1, 1) 
% 11. gallery(’randsvd’, N, 1e15, 4, 1, 1) 
% 12. same as #1, but a = a*1e-50 
% 13. gallery(’dorr’, N, 1e-4)
% 14. tridiag(a, 1e-8*ones(N,1), c) with a,c sampled from U (−1, 1) 
% 15. tridiag(a, zeros(N,1), c) with a,c sampled from U (−1, 1) 
% 16. tridiag(ones(N-1,1), 1e-8*ones(N,1),ones(N-1,1)) 
% 17. tridiag(ones(N-1,1), 1e8*ones(N,1),ones(N-1,1)) 
% 18. tridiag(-ones(N-1,1), 4*ones(N,1),-ones(N-1,1)) 
% 19. tridiag(-ones(N-1,1), 4*ones(N,1),ones(N-1,1)) 
% 20. tridiag(-ones(N-1,1), 4*ones(N,1),c), c sampled from U (−1, 1)

n = 512;

% Define true as a normal distribution of floating-point numbers with a mean value of 3 and standard deviation of 1.
xt = randn(n, 1) + 3;
writematrix(xt);

% keep condition number
fid = fopen("cond.dat", "w");

m1 = diag(rand(n - 1, 1) * 2 - 1, -1) + diag(rand(n, 1) * 2 - 1, 0) + diag(rand(n - 1, 1) * 2 - 1, 1);
eq1 = [m1 m1*xt];
fprintf(fid, "%e\n", cond(m1));
writematrix(eq1);

m2 = m1;
for i=1:n
    m2(i, i) = 1e8;
end
eq2 = [m2 m2*xt];
fprintf(fid, "%e\n", cond(m2));
writematrix(eq2);


m3 = gallery("lesp", n);
eq3 = [m3 m3*xt]
fprintf(fid, "%e\n", cond(m3));
writematrix(eq3);

m4 = m1;
m4(n/2+1, n/2) = m4(n/2+1, n/2) * 1e-50;
eq4 = [m4 m4*xt]
fprintf(fid, "%e\n", cond(m4));
writematrix(eq4);

m5 = m1;
for i=1:n-1
    if rand() < 0.5
        m5(i, i+1) = 0.0;
    end
end
for i=2:n
    if rand() < 0.5
        m5(i - 1, i) = 0.0;
    end
end
eq5 = [m5 m5*xt]
fprintf(fid, "%e\n", cond(m5));
writematrix(eq5)

m6 = 64 * eye(n) + diag(rand(n - 1, 1) * 2 - 1, -1) + diag(rand(n - 1, 1) * 2 - 1, 1);
eq6 = [m6 m6*xt]
fprintf(fid, "%e\n", cond(m6));
writematrix(eq6)

m7 = inv(gallery("kms", n, 0.5));
eq7 = [m7 m7*xt]
fprintf(fid, "%e\n", cond(m7));
writematrix(eq7);

m8 = gallery("randsvd", n, 1e15, 2, 1, 1);
eq8 = [m8 m8*xt]
fprintf(fid, "%e\n", cond(m8));
writematrix(eq8)
m9 = gallery("randsvd", n, 1e15, 3, 1, 1);
eq9 = [m9 m9*xt]
fprintf(fid, "%e\n", cond(m9));
writematrix(eq9)
m10 = gallery("randsvd", n, 1e15, 1, 1, 1);
eq10 = [m10 m10*xt]
fprintf(fid, "%e\n", cond(m10));
writematrix(eq10)
m11 = gallery("randsvd", n, 1e15, 4, 1, 1);
eq11 = [m11 m11*xt]
fprintf(fid, "%e\n", cond(m11));
writematrix(eq11)

m12 = m1;
for i=2:n
    m12(i - 1, i) = m12(i - 1, i) * 1e-50;
end
eq12 = [m12 m12*xt]
fprintf(fid, "%e\n", cond(m12));
writematrix(eq12)

m13 = gallery("dorr", n, 1e-4);
eq13 = [m13 m13*xt]
fprintf(fid, "%e\n", cond(m13));
writematrix(eq13)

% 14. tridiag(a, 1e-8*ones(N,1), c) with a,c sampled from U (−1, 1)
m14 = diag(rand(n - 1, 1) * 2 - 1, -1) + 1e-8 * eye(n) + diag(rand(n - 1, 1) * 2 - 1, 1);
eq14 = [m14 m14*xt]
fprintf(fid, "%e\n", cond(m14));
writematrix(eq14)

% 15. tridiag(a, zeros(N,1), c) with a,c sampled from U (−1, 1) 
m15 = diag(rand(n - 1, 1) * 2 - 1, -1) + zeros(n) + diag(rand(n - 1, 1) * 2 - 1, 1);
eq15 = [m15 m15*xt]
fprintf(fid, "%e\n", cond(m15));
writematrix(eq15)

% 16. tridiag(ones(N-1,1), 1e-8*ones(N,1),ones(N-1,1)) 
m16 = diag(ones(n - 1, 1), -1) + 1e-8 * eye(n) + diag(ones(n - 1, 1), 1);
eq16 = [m16 m16*xt]
fprintf(fid, "%e\n", cond(m16));
writematrix(eq16)

% 17. tridiag(ones(N-1,1), 1e8*ones(N,1),ones(N-1,1)) 
m17 = diag(ones(n - 1, 1), -1) + 1e8 * eye(n) + diag(ones(n - 1, 1), 1);
eq17 = [m17 m17*xt]
fprintf(fid, "%e\n", cond(m17));
writematrix(eq17)

% 18. tridiag(-ones(N-1,1), 4*ones(N,1),-ones(N-1,1)) 
m18 = diag(-1 * ones(n - 1, 1), -1) + 4 * eye(n) + diag(-1 * ones(n - 1, 1), 1);
eq18 = [m18 m18*xt]
fprintf(fid, "%e\n", cond(m18));
writematrix(eq18)

% 19. tridiag(-ones(N-1,1), 4*ones(N,1),ones(N-1,1)) 
m19 = diag(-1 * ones(n - 1, 1), -1) + 4 * eye(n) + diag(1 * ones(n - 1, 1), 1);
eq19 = [m19 m19*xt]
fprintf(fid, "%e\n", cond(m19));
writematrix(eq19)

% 20. tridiag(-ones(N-1,1), 4*ones(N,1),c), c sampled from U (−1, 1)
m20 = diag(-1 * ones(n - 1, 1), -1) + 4 * eye(n) + diag(rand(n - 1, 1) * 2 - 1, 1);
eq20 = [m20 m20*xt]
fprintf(fid, "%e\n", cond(m20));
writematrix(eq20)

fclose(fid);

