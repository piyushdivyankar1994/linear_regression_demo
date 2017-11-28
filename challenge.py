import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('challenge_dataset.txt', header=None)
x_val = df[[0]]
y_val = df[[1]]

# Train model on data
challenge_reg = linear_model.LinearRegression()
challenge_reg.fit(x_val, y_val)

# Print Score of the model
print challenge_reg.score(x_val, y_val)
print challenge_reg.coef_, challenge_reg.intercept_

print df

# Visulalize
plt.scatter(x_val, y_val)
plt.plot(x_val, challenge_reg.predict(x_val))
plt.show()

# To print the error of a particular value
# Replace 10 by any value of x
print challenge_reg.coef_* 10 +challenge_reg.intercept_

