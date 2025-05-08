import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

# Read the dataset
df = pd.read_csv("data.csv")

# Fill missing room values with the median of the room column
median_room = math.floor(df.room.median())
df.room = df.room.fillna(median_room)

# Define independent variables (X) and dependent variable (Y)
X = df[['area', 'room', 'age']]   # Independent variables
Y = df['price']                    # Dependent variable (target)

# Initialize and train the linear regression model
reg = linear_model.LinearRegression()
reg.fit(X, Y)

# Make predictions
y_pred = reg.predict(X)

# Print coefficients, intercept, and prediction for a new data point
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
print("Prediction for new data [2000 sq ft, 4 rooms, 10 years old]:", reg.predict([[2000, 4, 10]])[0])

# Plot the enhanced 3D graph
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual data, displaying all dimensions
sc = ax.scatter(df['area'], df['room'], df['price'], c=df['age'], cmap='viridis', s=100, alpha=0.9, edgecolors='black', label='Actual Data')

# Add color bar to represent "age"
cbar = plt.colorbar(sc)
cbar.set_label('Age (years)', fontsize=12)

# Create a mesh grid for the regression plane
area_range = np.linspace(df['area'].min(), df['area'].max(), 20)
room_range = np.linspace(df['room'].min(), df['room'].max(), 20)
area_mesh, room_mesh = np.meshgrid(area_range, room_range)

# Create a regression plane using the average age
age_mean = df['age'].mean()  # Average age for the plane
price_plane = reg.predict(np.c_[area_mesh.ravel(), room_mesh.ravel(), np.full(area_mesh.ravel().shape, age_mean)])
price_plane = price_plane.reshape(area_mesh.shape)

# Stylish regression plane with gradient color
ax.plot_surface(area_mesh, room_mesh, price_plane, cmap='plasma', alpha=0.6, edgecolor='none')

# Labels and legend
ax.set_xlabel('Area (sq ft)', fontsize=12, labelpad=15)
ax.set_ylabel('Rooms', fontsize=12, labelpad=15)
ax.set_zlabel('Price', fontsize=12, labelpad=15)
ax.set_title('Multiple Linear Regression', fontsize=18, fontweight='bold')

# Improve appearance
ax.view_init(elev=25, azim=135)  # Better viewing angle
ax.grid(True)

# Display the graph
plt.show()

# Display coefficients and predictions
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
print("Prediction for [2000 sq ft, 4 rooms, 10 years old]:", reg.predict([[2000, 4, 10]])[0])
