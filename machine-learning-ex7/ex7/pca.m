function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
covariance=1/m*(X'*X);
[U,S,V]=svd(covariance); % principle directions of data X (mxn) are the eigens vectors of cov'*cov (nxn * nxn)
%but covariance matrix is already symmetric, so eigen vectors of cov'*cov
%are eigen vectors of cov
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%







% =========================================================================

end
