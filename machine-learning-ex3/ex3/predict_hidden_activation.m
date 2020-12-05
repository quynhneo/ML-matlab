function p = predict_hidden_activation(Theta1, X)
%PREDICT Predict the max activation of the hidden layer of a trained neural network
%   p = PREDICT(Theta1, X) outputs the predicted label of max activation of the hidden layer given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%add ones to the data (the first feature x0 = 1)
X= [ones(size(X,1),1) X]; % 5000 x 401 (5000 examples, 401 features)
% propagate the variable to second layer
Z2= Theta1*X' ; % 25 variables x 401 features * 401 feature x5000 examples = 25 variables x 5000 example
% process
a2=sigmoid(Z2); % 25 x 5000

% find the label of the max probability labels between 1-25, return a 5000 x
% 1 vector of max activation.
[p_max,j]= max(a2,[],1); % j is 5000 by 1, index of max
% convert to column vector
p=j'; 


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
