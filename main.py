import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice
import plotly.graph_objects as go
from PIL import Image
from tqdm import tqdm


#Newton's Gravitational Constant
NGC = 6.67430e-11

#Load the ephemeris kernel
EPHEM_KERNEL_ADDRESS = "ephems/de440.bsp"
spice.furnsh(EPHEM_KERNEL_ADDRESS)
spice.furnsh("naif0012.tls") #This one handles the leapseconds
spice.furnsh("pck00010.tpc")

class CBody:
    def __init__(self, name, jpl_name, mass, radius):
        self.name       = name
        self.jpl_name   = jpl_name
        self.mass       = mass
        self.radius     = radius

        self.gparam     = self.mass*NGC

    def get_svec(self, et, frame="EARTH"):

        state, light_time = spice.spkezr(self.jpl_name, et, "J2000", "NONE", frame)

        #State Vector in meters instead of km
        state_m = state*1e3

        return state_m
    
class SCraft:
    def __init__(self, m_dry, m_prop, f_thrust, isp, svec_0):
        self.m_dry      = m_dry
        self.m_prop     = m_prop
        self.f_thrust   = f_thrust  #In N
        self.isp        = isp       #In s not m/s
        
        self.m_wet  = m_dry + m_prop
        self.mdot   = self.f_thrust / (self.isp * 9.81)

        self.dir    = "pro"
        self.att    = svec_0[3:]/np.linalg.norm(svec_0[3:])

        self.svec   = svec_0
        self.fire   = False


        self.m_prop_temp    = self.m_prop
        self.m_wet_temp     = self.m_wet
        self.svec_temp      = self.svec
        self.att_temp       = self.att

    def f_gravity(self, cbody_list, et, temp=False):
        f_g_sum = 0


        if not temp:
            m_use   = self.m_wet
            pos_sc  = self.svec[:3]
        else:
            m_use   = self.m_wet_temp
            pos_sc  = self.svec_temp[:3]

        for cbody in cbody_list:
            rvec = pos_sc - cbody.get_svec(et)[:3]
            rmag = np.linalg.norm(rvec)

            f_g_body = - (cbody.gparam * m_use * rvec)/(rmag**3)

            f_g_sum += f_g_body

        return f_g_sum


    def update(self, h, svec_new):
        if self.fire:
            delta_m     = - self.mdot * h
            self.m_prop += delta_m
            self.m_wet  += delta_m

        self.svec   = svec_new

        if self.dir == "pro":
            self.att = svec_new[3:]/np.linalg.norm(svec_new[3:])
        elif self.dir == "ret":
            self.att = - svec_new[3:]/np.linalg.norm(svec_new[3:])

        if self.m_prop < 0:
            self.fire = False

        self.m_prop_temp    = self.m_prop
        self.m_wet_temp     = self.m_wet
        self.svec_temp      = self.svec
        self.att_temp       = self.att


    def update_temp(self, h, svec_new):
        if self.fire:
            delta_m     = - self.mdot * h
            self.m_prop_temp = self.m_prop + delta_m
            self.m_wet_temp  = self.m_wet + delta_m

        self.svec_temp   = svec_new

        if self.dir == "pro":
            self.att_temp = svec_new[3:]/np.linalg.norm(svec_new[3:])
        elif self.dir == "ret":
            self.att_temp = - svec_new[3:]/np.linalg.norm(svec_new[3:])



class RK4Sim:
    def __init__(self, h, cbody_list, sc_list, t_start, t_end, center="EARTH"):
        self.h          = h
        self.cbody_list = cbody_list
        self.sc_list    = sc_list
        self.t_start    = t_start
        self.t_end      = t_end
        self.center     = center

        self.hcount = int((self.t_end - self.t_start)/h)

        self.pos_arr = np.zeros((self.hcount, 3))


    def tstep(self, sc_id, et):
        scraft = self.sc_list[sc_id]


        k1_v = self.h*scraft.svec[3:]
        k1_a = self.h*scraft.f_gravity(self.cbody_list, et)/scraft.m_wet


        svec_temp2 = np.zeros(6)
        svec_temp2[:3] = scraft.svec[:3] + k1_v/2
        svec_temp2[3:] = scraft.svec[3:] + k1_a/2
        scraft.update_temp(self.h/2, svec_temp2)

        k2_v = self.h*scraft.svec_temp[3:]
        k2_a = self.h*scraft.f_gravity(self.cbody_list, et+(self.h/2), temp=True)/scraft.m_wet_temp


        svec_temp3 = np.zeros(6)
        svec_temp3[:3] = scraft.svec[:3] + k2_v/2
        svec_temp3[3:] = scraft.svec[3:] + k2_a/2
        scraft.update_temp(self.h/2, svec_temp3)

        k3_v = self.h*scraft.svec_temp[3:]
        k3_a = self.h*scraft.f_gravity(self.cbody_list, et+(self.h/2), temp=True)/scraft.m_wet_temp


        svec_temp4 = np.zeros(6)
        svec_temp4[:3] = scraft.svec[:3] + k3_v
        svec_temp4[3:] = scraft.svec[3:] + k3_a
        scraft.update_temp(self.h/2, svec_temp4)

        k4_v = self.h*scraft.svec_temp[3:]
        k4_a = self.h*scraft.f_gravity(self.cbody_list, et+self.h, temp=True)/scraft.m_wet_temp


        svec_new     = np.zeros(6)
        svec_new[:3] = scraft.svec[:3] + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6
        svec_new[3:] = scraft.svec[3:] + (k1_a + 2*k2_a + 2*k3_a + k4_a)/6

        scraft.update(self.h, svec_new)

        return svec_new[:3]
    

    def run(self):
        et = self.t_start
        for i in tqdm(range(0, self.hcount)):
            self.pos_arr[i,:] = self.tstep(0, et)
            et += self.h

    def visualize(self,  animate_rate=50):

        # ---------------------------------------------------
        # Settings for animation
        # ---------------------------------------------------
        # Number of animation frames
        num_frames = 200  
        # Total simulation duration (in seconds)
        total_duration = self.t_end - self.t_start  
        # Time increment per animation frame (ephemeris time increment)
        dt_anim = total_duration / (num_frames - 1)
        # Map animation frame to index in self.pos_arr.
        pos_arr_indices = np.linspace(0, self.pos_arr.shape[0] - 1, num_frames).astype(int)

        # ---------------------------------------------------
        # Load the Earth texture image and prepare the sphere grid.
        # ---------------------------------------------------
        # Use a resolution that is a trade-off between quality and performance.
        res_theta, res_phi = 360, 180  # lower than 720x360 for smoother animation
        img = Image.open("earth.jpg")
        img = img.resize((res_theta, res_phi))
        img_np = np.array(img)

        # Create spherical coordinate grid.
        theta = np.linspace(0, 2 * np.pi, res_theta, endpoint=False)
        phi = np.linspace(0, np.pi, res_phi)
        theta, phi = np.meshgrid(theta, phi)

        # Cartesian coordinates for a sphere scaled by Earth’s radius.
        # These coordinates are in the inertial frame.
        # (We will rotate them for each frame.)
        # First, identify the central body (assumed Earth) to get its radius.
        earth = None
        for body in self.cbody_list:
            if body.name.upper() == self.center.upper():
                earth = body
                break
        if earth is None:
            raise ValueError("Central body '{}' not found".format(self.center))
        radius = earth.radius

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        # Flattened arrays for Mesh3d.
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        # Build the vertex colors (from the image) for the Mesh3d.
        vertex_colors = []
        for i in range(res_phi):
            for j in range(res_theta):
                r, g, b = img_np[i, j][:3]
                vertex_colors.append(f"rgb({r},{g},{b})")

        # Build mesh faces (triangles) for the Earth.
        faces = []
        for i in range(res_phi - 1):
            for j in range(res_theta):
                next_j = (j + 1) % res_theta
                idx1 = i * res_theta + j
                idx2 = i * res_theta + next_j
                idx3 = (i + 1) * res_theta + j
                idx4 = (i + 1) * res_theta + next_j
                faces.append([idx1, idx3, idx2])
                faces.append([idx2, idx3, idx4])
        faces = np.array(faces)
        i_faces = faces[:, 0]
        j_faces = faces[:, 1]
        k_faces = faces[:, 2]

        # Pre-calculate the base atmosphere mesh (a slightly larger sphere).
        scale_atmo = 1.03  # Atmosphere 3% larger than Earth.
        x_atmo = scale_atmo * x
        y_atmo = scale_atmo * y
        z_atmo = scale_atmo * z

        # ---------------------------------------------------
        # Prepare animation frames.
        # ---------------------------------------------------
        frames = []
        for frame_idx in range(num_frames):
            # Compute current epoch for this frame.
            et_current = self.t_start + frame_idx * dt_anim
            # Get the rotation matrix from "J2000" to "IAU_EARTH" at et_current.
            rot_mat = spice.pxform("J2000", "IAU_EARTH", et_current)

            # Rotate the Earth sphere vertices.
            coords = np.vstack((x_flat, y_flat, z_flat))
            rotated_coords = rot_mat.dot(coords)
            x_rot = rotated_coords[0]
            y_rot = rotated_coords[1]
            z_rot = rotated_coords[2]

            # Rotate the atmospheric layer vertices.
            coords_atmo = np.vstack((x_atmo.flatten(), y_atmo.flatten(), z_atmo.flatten()))
            rotated_atmo = rot_mat.dot(coords_atmo)
            x_atmo_rot = rotated_atmo[0].reshape(x.shape)
            y_atmo_rot = rotated_atmo[1].reshape(x.shape)
            z_atmo_rot = rotated_atmo[2].reshape(x.shape)

            # Process the orbit data.
            # We'll display the orbit path from the beginning up to the current animation index.
            idx = pos_arr_indices[frame_idx]
            orbit_segment = self.pos_arr[:idx + 1]
            # Rotate the orbit segment using the same rotation.
            orbit_rot = []
            for pos in orbit_segment:
                orbit_rot.append(rot_mat.dot(pos))
            orbit_rot = np.array(orbit_rot)

            # Create updated trace dictionaries for this frame.
            frame_data = [
                # Earth Mesh3d (textured globe).
                dict(
                    type="mesh3d",
                    x=x_rot,
                    y=y_rot,
                    z=z_rot,
                    i=i_faces,
                    j=j_faces,
                    k=k_faces,
                    vertexcolor=vertex_colors,
                    flatshading=True,
                    lighting=dict(ambient=1.0, diffuse=1.0, specular=0.5)
                ),
                # Atmosphere Surface.
                dict(
                    type="surface",
                    x=x_atmo_rot,
                    y=y_atmo_rot,
                    z=z_atmo_rot,
                    colorscale=[[0, "rgba(135,206,235,0.3)"], [1, "rgba(135,206,235,0.3)"]],
                    showscale=False,
                    opacity=0.5,
                    lighting=dict(ambient=1.0)
                ),
                # Orbit trace (line up to current point).
                dict(
                    type="scatter3d",
                    mode="lines+markers",
                    x=orbit_rot[:, 0] if orbit_rot.size else [],
                    y=orbit_rot[:, 1] if orbit_rot.size else [],
                    z=orbit_rot[:, 2] if orbit_rot.size else [],
                    line=dict(color="red", width=4),
                    marker=dict(color="red", size=4),
                )
            ]

            frames.append(dict(data=frame_data, name=str(frame_idx)))

        # ---------------------------------------------------
        # Build the initial traces using the first frame.
        # ---------------------------------------------------
        # Use et = self.t_start for the initial frame.
        et0 = self.t_start
        rot_mat0 = spice.pxform("J2000", "IAU_EARTH", et0)
        coords0 = np.vstack((x_flat, y_flat, z_flat))
        rotated_coords0 = rot_mat0.dot(coords0)
        x_rot0 = rotated_coords0[0]
        y_rot0 = rotated_coords0[1]
        z_rot0 = rotated_coords0[2]

        coords_atmo0 = np.vstack((x_atmo.flatten(), y_atmo.flatten(), z_atmo.flatten()))
        rotated_atmo0 = rot_mat0.dot(coords_atmo0)
        x_atmo_rot0 = rotated_atmo0[0].reshape(x.shape)
        y_atmo_rot0 = rotated_atmo0[1].reshape(x.shape)
        z_atmo_rot0 = rotated_atmo0[2].reshape(x.shape)

        # For the orbit, show the starting point only.
        orbit0 = self.pos_arr[:1]
        orbit_rot0 = []
        for pos in orbit0:
            orbit_rot0.append(rot_mat0.dot(pos))
        orbit_rot0 = np.array(orbit_rot0)

        # Create the initial traces.
        earth_trace = go.Mesh3d(
            x=x_rot0,
            y=y_rot0,
            z=z_rot0,
            i=i_faces,
            j=j_faces,
            k=k_faces,
            vertexcolor=vertex_colors,
            flatshading=True,
            lighting=dict(ambient=1.0, diffuse=1.0, specular=0.5),
            name="Earth"
        )
        atmosphere_trace = go.Surface(
            x=x_atmo_rot0,
            y=y_atmo_rot0,
            z=z_atmo_rot0,
            colorscale=[[0, "rgba(135,206,235,0.3)"], [1, "rgba(135,206,235,0.3)"]],
            showscale=False,
            opacity=0.5,
            lighting=dict(ambient=1.0),
            name="Atmosphere"
        )
        orbit_trace = go.Scatter3d(
            x=orbit_rot0[:, 0],
            y=orbit_rot0[:, 1],
            z=orbit_rot0[:, 2],
            mode="lines+markers",
            line=dict(color="red", width=4),
            marker=dict(color="red", size=4),
            name="Orbit"
        )

        # ---------------------------------------------------
        # Build the figure with frames and animation settings.
        # ---------------------------------------------------
        fig = go.Figure(
            data=[earth_trace, atmosphere_trace, orbit_trace],
            frames=frames
        )

        # Add play and slider buttons.
        fig.update_layout(
            title="Animated Earth Rotation with Spacecraft Orbit",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="black",
                aspectmode="data"
            ),
            plot_bgcolor="black",
            paper_bgcolor="black",
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": animate_rate, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                }]
            }],
            sliders=[{
                "steps": [{
                    "args": [[str(k)], {"frame": {"duration": animate_rate, "redraw": True}, "mode": "immediate"}],
                    "label": str(k),
                    "method": "animate"
                } for k in range(num_frames)],
                "transition": {"duration": 0},
                "x": 0.1,
                "y": 0,
                "currentvalue": {"font": {"size": 12}, "prefix": "Frame: ", "visible": True},
                "len": 0.9
            }]
        )

        fig.show()




# Define Earth and spacecraft, then run the simulation
earth = CBody("Earth", "EARTH", 5.9722e24, 6378137)
# Note: The spacecraft state vector should contain position (first three values)
# and velocity (last three values). Here we assume the units are consistent (meters, m/s).
sat1 = SCraft(100, 80, 1000, 320, np.array([0, earth.radius + 400000, 0, 7000, 0, 1200]))
sim1 = RK4Sim(0.1, [earth], [sat1], 0, 4000)
sim1.run()
sim1.visualize()
