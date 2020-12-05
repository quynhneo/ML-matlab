%% =============== Part 4: Loading and Visualizing Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('ex7faces.mat') % the 5000 faces data
face=imread('bw.jpg');
face=im2double(face);


face=reshape(face,[1,prod(size(face))] );

%displayData(face);


fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this might take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%renorm the face
%face_norm = bsxfun(@minus, face, mu);
%face_norm = bsxfun(@rdivide, face_norm, sigma);
f2 = figure;
displayData(face);

%  Run PCA
[U, S] = pca(X_norm); % columns of U are eigenvectors of the covariance matrix .
% element ij of covariance matrix is the ensemble average over 5000
% pictures of pixel i and j

%  Visualize the top 4 eigenvectors found
displayData(U(:, 1:4)');
hold on;


fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Part 6: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');


Z =projectData(face,U,3);


fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

face_rec =(Z.*U(:,1:3))';

% Display normalized data
subplot(2, 2, 1);
displayData(face);
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(2, 2, 2);
displayData(face_rec(1,:));
title(Z(:,1));
axis square;


subplot(2, 2, 3);
displayData(face_rec(2,:));
title(Z(:,2));
axis square;

subplot(2, 2, 4);
displayData(face_rec(3,:));
title(Z(:,3));
axis square;


fprintf('Program paused. Press enter to continue.\n');
pause;
