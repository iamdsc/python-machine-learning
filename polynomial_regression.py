# Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


np.random.seed(2)
pageSpeeds = np.random.normal(3,1,1000)
purchaseAmount = np.random.normal(50,10,1000) / pageSpeeds

x = np.array(pageSpeeds)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x, y, 4))

# plotting the data and best fit line
xp = np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()

# calculating r-squared value
r2 = r2_score(y, p4(x))
print(r2)
