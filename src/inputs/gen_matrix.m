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
% keep condition number
fid = fopen("cond.dat", "w");

m1 = diag(rand(n - 1, 1) * 2 - 1, -1) + diag(rand(n, 1) * 2 - 1, 0) + diag(rand(n - 1, 1) * 2 - 1, 1);
fprintf(fid, "%e\n", cond(m1));
save('m1.txt', 'm1', '-ascii');

m2 = m1;
for i=1:n
    m2(i, i) = 1e8;
end
fprintf(fid, "%e\n", cond(m2));
save('m2.txt', 'm2', '-ascii');

m3 = [gallery("lesp", n), rand(n, 1) * 2 - 1];
fprintf(fid, "%e\n", cond(m3));
save('m3.txt', 'm3', '-ascii');

m4 = m1;
m4(n/2+1, n/2) = m4(n/2+1, n/2) * 1e-50;
fprintf(fid, "%e\n", cond(m4));
save('m4.txt', 'm4', '-ascii');

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
fprintf(fid, "%e\n", cond(m5));
save('m5.txt', 'm5', '-ascii')

m6 = 64 * eye(n) + diag(rand(n - 1, 1) * 2 - 1, -1) + diag(rand(n - 1, 1) * 2 - 1, 1);
fprintf(fid, "%e\n", cond(m6));
save('m6.txt', 'm6', '-ascii')

m7 = inv(gallery("kms", n, 0.5));
fprintf(fid, "%e\n", cond(m7));
save('m7.txt', 'm7', '-ascii');

m8 = gallery("randsvd", n, 1e15, 2, 1, 1);
fprintf(fid, "%e\n", cond(m8));
save('m8.txt', 'm8', '-ascii')
m9 = gallery("randsvd", n, 1e15, 3, 1, 1);
fprintf(fid, "%e\n", cond(m9));
save('m9.txt', 'm9', '-ascii')
m10 = gallery("randsvd", n, 1e15, 1, 1, 1);
fprintf(fid, "%e\n", cond(m10));
save('m10.txt', 'm10', '-ascii')
m11 = gallery("randsvd", n, 1e15, 4, 1, 1);
fprintf(fid, "%e\n", cond(m11));
save('m11.txt', 'm11', '-ascii')

m12 = m1;
for i=2:n
    m12(i - 1, i) = m12(i - 1, i) * 1e-50;
end
fprintf(fid, "%e\n", cond(m12));
save('m12.txt', 'm12', '-ascii')

m13 = gallery("dorr", n, 1e-4);
fprintf(fid, "%e\n", cond(m13));
save('m13.txt', 'm13', '-ascii')

% 14. tridiag(a, 1e-8*ones(N,1), c) with a,c sampled from U (−1, 1)
m14 = diag(rand(n - 1, 1) * 2 - 1, -1) + 1e-8 * eye(n) + diag(rand(n - 1, 1) * 2 - 1, 1);
fprintf(fid, "%e\n", cond(m14));
save('m14.txt', 'm14', '-ascii')

% 15. tridiag(a, zeros(N,1), c) with a,c sampled from U (−1, 1) 
m15 = diag(rand(n - 1, 1) * 2 - 1, -1) + zeros(n) + diag(rand(n - 1, 1) * 2 - 1, 1);
fprintf(fid, "%e\n", cond(m15));
save('m15.txt', 'm15', '-ascii')

% 16. tridiag(ones(N-1,1), 1e-8*ones(N,1),ones(N-1,1)) 
m16 = diag(ones(n - 1, 1), -1) + 1e-8 * eye(n) + diag(ones(n - 1, 1), 1);
fprintf(fid, "%e\n", cond(m16));
save('m16.txt', 'm16', '-ascii')

% 17. tridiag(ones(N-1,1), 1e8*ones(N,1),ones(N-1,1)) 
m17 = diag(ones(n - 1, 1), -1) + 1e8 * eye(n) + diag(ones(n - 1, 1), 1);
fprintf(fid, "%e\n", cond(m17));
save('m17.txt', 'm17', '-ascii')

% 18. tridiag(-ones(N-1,1), 4*ones(N,1),-ones(N-1,1)) 
m18 = diag(-1 * ones(n - 1, 1), -1) + 4 * eye(n) + diag(-1 * ones(n - 1, 1), 1);
fprintf(fid, "%e\n", cond(m18));
save('m18.txt', 'm18', '-ascii')

% 19. tridiag(-ones(N-1,1), 4*ones(N,1),ones(N-1,1)) 
m19 = diag(-1 * ones(n - 1, 1), -1) + 4 * eye(n) + diag(1 * ones(n - 1, 1), 1);
fprintf(fid, "%e\n", cond(m19));
save('m19.txt', 'm19', '-ascii')

% 20. tridiag(-ones(N-1,1), 4*ones(N,1),c), c sampled from U (−1, 1)
m20 = diag(-1 * ones(n - 1, 1), -1) + 4 * eye(n) + diag(rand(n - 1, 1) * 2 - 1, 1);
fprintf(fid, "%e\n", cond(m20));
save('m20.txt', 'm20', '-ascii')

fclose(fid);

