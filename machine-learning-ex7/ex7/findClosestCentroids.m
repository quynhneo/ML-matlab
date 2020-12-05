function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); % 3

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);%300 x 1
D=zeros(size(X,1),K); %300x3
for i=1:K
    differences=X-centroids(i,:);% coordinate difference , each data point - centroid i,  300x2
    distances=sum(differences.^2,2); % collapse to get the distance (square), 300x1
    D(:,i)=distances;%save the distances sqrt to centroid 
end
[m,idx]=min(D,[],2);% find the min distance sqrt m, and index

    
    
    


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%







% =============================================================

end

