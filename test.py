import numpy as np
import plotly.graph_objects as go
from PIL import Image

# Set resolution for the Earth mesh.
res_theta, res_phi = 720, 360

# Load and resize the Earth image.
img = Image.open("earth.jpg")
img = img.resize((res_theta, res_phi))
img_np = np.array(img)

# Create a grid in spherical coordinates:
theta = np.linspace(0, 2 * np.pi, res_theta, endpoint=False)
phi = np.linspace(0, np.pi, res_phi)
theta, phi = np.meshgrid(theta, phi)

# Cartesian coordinates for the Earth sphere (unit sphere)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Flatten arrays for the Mesh3d trace.
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

# Map each vertex to its corresponding pixel color from the image.
vertex_colors = []
for i in range(res_phi):
    for j in range(res_theta):
        r, g, b = img_np[i, j][:3]
        vertex_colors.append(f"rgb({r},{g},{b})")

# Build mesh faces for the Earth.
faces = []
for i in range(res_phi - 1):
    for j in range(res_theta):
        next_j = (j + 1) % res_theta
        idx1 = i * res_theta + j
        idx2 = i * res_theta + next_j
        idx3 = (i + 1) * res_theta + j
        idx4 = (i + 1) * res_theta + next_j
        faces.append([idx1, idx3, idx2])  # First triangle
        faces.append([idx2, idx3, idx4])  # Second triangle

faces = np.array(faces)
i_faces = faces[:, 0]
j_faces = faces[:, 1]
k_faces = faces[:, 2]

# Create the Earth Mesh3d.
earth_trace = go.Mesh3d(
    x=x_flat,
    y=y_flat,
    z=z_flat,
    i=i_faces,
    j=j_faces,
    k=k_faces,
    vertexcolor=vertex_colors,
    flatshading=True,
    lighting=dict(ambient=1.0, diffuse=1.0, specular=0.5)
)

# Create an atmosphere layer:
scale = 1.03  # Atmosphere is 3% larger than the Earth sphere.
x_atmo = scale * x
y_atmo = scale * y
z_atmo = scale * z

atmosphere_trace = go.Surface(
    x=x_atmo,
    y=y_atmo,
    z=z_atmo,
    colorscale=[[0, "rgba(135,206,235,0.3)"], [1, "rgba(135,206,235,0.3)"]],
    showscale=False,
    opacity=0.5,
    lighting=dict(ambient=1.0)
)

# Combine both traces in the same figure.
fig = go.Figure(data=[earth_trace, atmosphere_trace])

# Update layout for a black background.
fig.update_layout(
    title="Earth with Atmospheric Glow",
    plot_bgcolor="black",
    paper_bgcolor="black",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="black"  # 3D scene background color
    )
)

fig.show()
