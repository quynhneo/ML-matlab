function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params,num_labels, (input_layer_size + 1));

% Setup some useful variables
m = size(X, 1); 
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X=[ones(size(X,1),1),X]; %add a column of 1 as the first feature for every example
for i=1:m % for every row
    % forward feed
    a1=X(i,:); %  1 by 401
    z2 = Theta1* a1'; %  25 x401 act on 401x1 -> 25 x 1
    a2 = sigmoid(z2); %25x1

    
    %convert the correct output y to a vector
    w = zeros(size(Theta1,1),1); %10x1
    w(y(i)+1) = 1;
    
    %sum of cost over output for i example
    J= J + -w' * log(a2) - (1.- w')*log(1.-a2);
    
    %back propagation
    delta2 = a2 - w; % 10x1
   
    Theta1_grad = Theta1_grad + delta2*a1;  % 25x1*1x401 -> 25 x 401: grad J wrt theta1
   

end

%add regularization
% remove the first column of each theta matrix (the bias term)
%add the sum square of every elements
J =J/m+ lambda/2/m* sum(sum(Theta1(:,2:size(Theta1,2)).^2));
Theta1_grad = Theta1_grad./m;


%REGULARIZE
Theta1_grad(:,2:size(Theta1_grad,2))=Theta1_grad(:,2:size(Theta1_grad,2))+ lambda/m.*Theta1(:,2:size(Theta1,2));
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = Theta1_grad(:);


end
