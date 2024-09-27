import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize parameters
num_points = 100  # Number of points to display
x_data = np.arange(num_points)
y_data = np.zeros(num_points)

# Create a figure and axis
fig, ax = plt.subplots()
line, = ax.plot(x_data, y_data)

# Set up the plot limits
ax.set_ylim(0, 1)
ax.set_xlim(0, num_points)

# Function to update the plot
def update(frame):
    # Shift the y_data to the left and add new random data
    y_data[:-1] = y_data[1:]
    y_data[-1] = np.random.rand()  # Generate a new random data point
    line.set_ydata(y_data)  # Update the data of the line
    return line,

# Create an animation that updates every 100 ms
ani = animation.FuncAnimation(fig, update, blit=True, interval=20)

# Show the plot
plt.show()
