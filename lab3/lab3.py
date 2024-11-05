import numpy as np
import matplotlib.pyplot as plt

# Input values
x0 = 18  # Ensure x0 is defined before usage
y0 = 24  # Ensure y0 is defined before usage

# Define equations of the membership functions using line equations

# Membership Function A1 (Trapezoidal)
def mu_A1(x):
    if 16 <= x <= 20:
        return (x - 16) / 4
    elif 20 < x <= 30:
        return 1
    elif 30 < x <= 35:
        return -(x - 30) / 5 + 1
    else:
        return 0

# Membership Function A2 (Triangular)
def mu_A2(x):
    if 15 <= x <= 20:
        return (x - 15) / 5
    elif 20 < x <= 28:
        return -(x - 20) / 8 + 1
    else:
        return 0

# Membership Function B1 (Triangular)
def mu_B1(y):
    if 18 <= y <= 23:
        return (y - 18) / 5
    elif 23 < y <= 30:
        return -(y - 23) / 7 + 1
    else:
        return 0

# Membership Function B2 (Trapezoidal)
def mu_B2(y):
    if 21 <= y <= 26:
        return (y - 21) / 5
    elif 26 < y <= 32:
        return 1
    elif 32 < y <= 40:
        return -(y - 32) / 8 + 1
    else:
        return 0

# Calculate membership degrees at x0 and y0
mu_A1_x0 = mu_A1(x0)
mu_A2_x0 = mu_A2(x0)
mu_B1_y0 = mu_B1(y0)
mu_B2_y0 = mu_B2(y0)

# Print membership degrees
print(f"Membership degrees at x0 = {x0}:")
print(f"μ_A1({x0}) = {mu_A1_x0}")
print(f"μ_A2({x0}) = {mu_A2_x0}")

print(f"\nMembership degrees at y0 = {y0}:")
print(f"μ_B1({y0}) = {mu_B1_y0}")
print(f"μ_B2({y0}) = {mu_B2_y0}")

# Calculate firing strengths
alpha1 = min(mu_A1_x0, mu_B1_y0)
alpha2 = min(mu_A2_x0, mu_B2_y0)

# Print firing strengths
print(f"\nFiring strengths:")
print(f"α1 = min({mu_A1_x0}, {mu_B1_y0}) = {alpha1}")
print(f"α2 = min({mu_A2_x0}, {mu_B2_y0}) = {alpha2}")

# Coefficients
a1, b1 = 0.6, 0.8
a2, b2 = 0.3, 0.2

# Calculate rule outputs
z1 = a1 * x0 + b1 * y0
z2 = a2 * x0 + b2 * y0

# Print rule outputs
print(f"\nRule outputs:")
print(f"z1 = {a1} * {x0} + {b1} * {y0} = {z1}")
print(f"z2 = {a2} * {x0} + {b2} * {y0} = {z2}")

# Calculate crisp output z0
numerator = alpha1 * z1 + alpha2 * z2
denominator = alpha1 + alpha2
z0 = numerator / denominator

# Print crisp output
print(f"\nCrisp output z0:")
print(f"z0 = ({alpha1} * {z1} + {alpha2} * {z2}) / ({alpha1} + {alpha2}) = {z0}")

# Plot membership functions
# x-axis values for plotting
x_values = np.linspace(10, 40, 300)
y_values = np.linspace(15, 45, 300)

# Calculate membership values for plotting
A1_mf = [mu_A1(x) for x in x_values]
A2_mf = [mu_A2(x) for x in x_values]
B1_mf = [mu_B1(y) for y in y_values]
B2_mf = [mu_B2(y) for y in y_values]

# Plot A1 and A2 membership functions
plt.figure(figsize=(10, 5))
plt.plot(x_values, A1_mf, label='A1')
plt.plot(x_values, A2_mf, label='A2')
plt.scatter([x0], [mu_A1_x0], color='red', label='μ_A1(x0)')
plt.scatter([x0], [mu_A2_x0], color='green', label='μ_A2(x0)')
plt.title('Membership Functions for A1 and A2')
plt.xlabel('x')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)
plt.show()

# Plot B1 and B2 membership functions
plt.figure(figsize=(10, 5))
plt.plot(y_values, B1_mf, label='B1')
plt.plot(y_values, B2_mf, label='B2')
plt.scatter([y0], [mu_B1_y0], color='red', label='μ_B1(y0)')
plt.scatter([y0], [mu_B2_y0], color='green', label='μ_B2(y0)')
plt.title('Membership Functions for B1 and B2')
plt.xlabel('y')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)
plt.show()
