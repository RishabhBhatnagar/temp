from ml_csdlo6021.multivariate_linear_regression import LinearRegressor


data_set = [[2, i, 2*i] for i in range(10)]  # table of 2.
regressor = LinearRegressor(data_set)

regressor.fit(α=0.009, max_epochs=200)

test_data = [[2, 11], [2, 12], [2, 13], [2, 14]]
target_data = [22, 24, 26, 28]

predictions = regressor.predict(test_data)
print(regressor.accuracy(predictions, target_data), "is the accuracy")


'''
Output:
1) with learning rate = 1.0
python3.5/site-packages/ml_csdlo6021/multivariate_linear_regression.py:33: RuntimeWarning: overflow encountered in double_scalars
  for i in range(self.m)]
200 epochs elapsed
-5.381863567960958e+239 is the current accuracy.
Do you want to stop training (y/*)??n
python3.5/site-packages/ml_csdlo6021/multivariate_linear_regression.py:41: RuntimeWarning: overflow encountered in double_scalars
  for i in range(self.m)
Something bad happened while training.
 Try fitting with new learning rate.
nan is the accuracy


2) with learning_rate = 0.1
200 epochs elapsed
0.9857607342477104 is the current accuracy.
Do you want to stop training (y/*)??n
400 epochs elapsed
0.9983464828066766 is the current accuracy.
Do you want to stop training (y/*)??n
600 epochs elapsed
0.9998079873529873 is the current accuracy.
Do you want to stop training (y/*)??n
800 epochs elapsed
0.9999777027679169 is the current accuracy.
Do you want to stop training (y/*)??y
Training aborted.

0.999969727374987 is the accuracy
'''
