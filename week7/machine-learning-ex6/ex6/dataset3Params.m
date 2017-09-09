function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

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

Cvec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaVec = [0.01 0.03 0.1 0.3 1 3 10 30];

cvError = 100000;
Ci = 1;
Si = 1;

for i = 1:size(Cvec,2),
  C = Cvec(i);
  for j = 1:size(sigmaVec,2),
    sigma = sigmaVec(j);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    errors = mean(double(predictions ~= yval));
    if cvError > errors,
      cvError = errors;
      Ci = i;
      Si = j;
    end;
  end;
end;

C = Cvec(Ci);
sigma = sigmaVec(Si);

fprintf('C = %d, simga = %d', C, sigma);




% =========================================================================

end
