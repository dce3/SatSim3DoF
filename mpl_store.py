# Set simulation parameters: timestep of 0.1 seconds, running from t=0 to t=20 seconds.

print(sim1.pos_arr)  # Check the recorded state array

# Extract the position data (assuming pos_arr stores position data)
x = sim1.pos_arr[::10, 0]
y = sim1.pos_arr[::10, 1]
z = sim1.pos_arr[::10, 2]

# Create a 3D plot for the orbit trajectory
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')



# Generate a mesh for the Earth sphere using spherical coordinates
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = earth.radius * np.outer(np.cos(u), np.sin(v))
y_sphere = earth.radius * np.outer(np.sin(u), np.sin(v))
z_sphere = earth.radius * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the Earth as a semi-transparent sphere
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=1, rstride=4, cstride=4)

# Label axes (units in meters)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# Calculate the midpoints and maximum range for equal axes
max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
mid_x = (x.max() + x.min()) * 0.5
mid_y = (y.max() + y.min()) * 0.5
mid_z = (z.max() + z.min()) * 0.5

# Set the axes limits so that all axes have the same range
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# For Matplotlib 3.3 and later, you can also force equal aspect ratio:
ax.set_box_aspect([1, 1, 1])

# Optionally, display a legend
ax.legend()

# Plot the orbit trajectory with markers
ax.plot(x, y, z, label='Orbit Trajectory', c="red")

# Show the final plot
plt.show()