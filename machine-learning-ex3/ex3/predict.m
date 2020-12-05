function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%add ones to the data (the first feature x0 = 1)
X= [ones(size(X,1),1) X]; % 5000 x 401 (5000 examples, 401 features)
% propagate the variable to second layer
Z2= Theta1*X' ; % 25 variables x 401 features * 401 feature x5000 examples = 25 variables x 5000 example
% process
a2=sigmoid(Z2); % 25 x 5000
% add the 0 feature
a2=[ones(1,size(a2,2)) ;a2]; % 26 x 5000
%propagate to the third layer
Z3 = Theta2*a2 ; % 10x26 times 26 x 5000 = 10 x 5000
% process
a3 = sigmoid(Z3); % 10 x 5000
% find the label of the max probability labels between 10, return a 5000 x
% 1 vector of predicted labels for 5000 training examples
[p_max,j]= max(a3,[],1);
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
