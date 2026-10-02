import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(true_values, predicted_values, alpha=0.5)
plt.plot(
    [true_values.min(), true_values.max()],
    [true_values.min(), true_values.max()],
    'r--',
)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()