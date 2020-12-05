function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];

error_matrix = ones(length(C_list),length(sigma_list));% row of C, column of sigma

for i=1:length(C_list)
    for j=1:length(sigma_list)
        %TRAIN ON THE TRAINING SET
        model= svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j)));
        %TEST ON THE CROSS VALIDATION SET
        predictions = svmPredict(model,Xval);
        error_matrix(i,j)=mean(double(predictions ~= yval));
        
    end
end

[aa bb] = min(error_matrix); % min of each column, arrange in a row. n is the row index
[ll kk ] = min(aa); % min of m, k is the column index
C = C_list(bb(kk))
sigma= sigma_list(kk)

        

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
