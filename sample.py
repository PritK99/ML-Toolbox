import matplotlib.pyplot as plt

# Example runtime data (replace these with your actual data)
dimensions = [4, 5, 6, 7, 8, 9, 10]
runtime_algorithm_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # only till 6x6
runtime_algorithm_2 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
runtime_algorithm_3 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # proposed algorithm

# Plotting
plt.figure(figsize=(8, 6))

# Plot for Algorithm 1 (up to 6x6)
plt.plot(dimensions[:6], runtime_algorithm_1, label='Algorithm 1', color='blue', marker='o')

# Plot for Algorithm 2 (all dimensions)
plt.plot(dimensions, runtime_algorithm_2, label='Algorithm 2', color='red', marker='s')

# Plot for Algorithm 3 (all dimensions, green for proposed algorithm)
plt.plot(dimensions, runtime_algorithm_3, label='Proposed Algorithm', color='green', marker='^')

# Adding titles and labels
plt.title('Runtime Comparison of Algorithms')
plt.xlabel('Dimension (n x n)')
plt.ylabel('Runtime (seconds)')
plt.xticks(dimensions)  # ensures each dimension value appears on the x-axis
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
