%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% map feature
X = mapFeature(X(:,1), X(:,2));




%% Setup the parameters you will use for this exercise
%
input_layer_size  = size(X,2) ;  % 
 
num_labels = 2;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

                         
%% weight matrix

%
Theta1=randn(num_labels, size(X,2)+1);

%% 

%%compute cost 
% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(Theta1(:), input_layer_size,num_labels, X, y, lambda);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 
%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[Theta1, cost] = fmincg(costFunction, Theta1(:), options);

Theta1 = reshape(Theta1,num_labels, (input_layer_size + 1));



fprintf('Program paused. Press enter to continue.\n');
pause;
%% %% ================= Part 3: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the trainingo set accuracy.

pred = predict(Theta1, X);
%pause;
fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.


%% 
% Plot Boundary
X_w_one=[ones(size(X,1),1),X];
plotDecisionBoundary(Theta1(1,:), X_w_one, y);
plotDecisionBoundary(Theta1(2,:), X_w_one, y);




