import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import spiceypy as spice
from tqdm import tqdm

# Newton's Gravitational Constant
NGC = 6.67430e-11

# Load the ephemeris kernel
EPHEM_KERNEL_ADDRESS = "ephems/de440.bsp"
spice.furnsh(EPHEM_KERNEL_ADDRESS)
spice.furnsh("naif0012.tls")  # Handles leap seconds


class CBody:
    def __init__(self, name, jpl_name, mass, radius, color="white"):
        self.name = name
        self.jpl_name = jpl_name
        self.mass = mass
        self.radius = radius
        self.color = color

        self.gparam = self.mass * NGC

    def get_svec(self, et, frame="SUN"):
        state, light_time = spice.spkezr(self.jpl_name, et, "J2000", "NONE", frame)
        state_m = state * 1e3  # Convert km to m
        return state_m


class SCraft:
    def __init__(self, m_dry, m_prop, f_thrust, isp, svec_0):
        self.m_dry = m_dry
        self.m_prop = m_prop
        self.f_thrust = f_thrust  # In N
        self.isp = isp  # In s
        self.m_wet = m_dry + m_prop
        self.mdot = self.f_thrust / (self.isp * 9.81)

        self.dir = "pro"
        self.att = svec_0[3:] / np.linalg.norm(svec_0[3:])
        self.svec = svec_0
        self.fire = False

        self.m_prop_temp = self.m_prop
        self.m_wet_temp = self.m_wet
        self.svec_temp = self.svec
        self.att_temp = self.att

    def f_gravity(self, cbody_list, et, temp=False):
        f_g_sum = 0
        if not temp:
            m_use = self.m_wet
            pos_sc = self.svec[:3]
        else:
            m_use = self.m_wet_temp
            pos_sc = self.svec_temp[:3]

        for cbody in cbody_list:
            rvec = pos_sc - cbody.get_svec(et)[:3]
            rmag = np.linalg.norm(rvec)
            f_g_body = - (cbody.gparam * m_use * rvec) / (rmag**3)
            f_g_sum += f_g_body

        return f_g_sum

    def update(self, h, svec_new):
        if self.fire:
            delta_m = -self.mdot * h
            self.m_prop += delta_m
            self.m_wet += delta_m

        self.svec = svec_new

        if self.dir == "pro":
            self.att = svec_new[3:] / np.linalg.norm(svec_new[3:])
        elif self.dir == "ret":
            self.att = -svec_new[3:] / np.linalg.norm(svec_new[3:])

        if self.m_prop < 0:
            self.fire = False

        self.m_prop_temp = self.m_prop
        self.m_wet_temp = self.m_wet
        self.svec_temp = self.svec
        self.att_temp = self.att

    def update_temp(self, h, svec_new):
        if self.fire:
            delta_m = -self.mdot * h
            self.m_prop_temp = self.m_prop + delta_m
            self.m_wet_temp = self.m_wet + delta_m

        self.svec_temp = svec_new

        if self.dir == "pro":
            self.att_temp = svec_new[3:] / np.linalg.norm(svec_new[3:])
        elif self.dir == "ret":
            self.att_temp = -svec_new[3:] / np.linalg.norm(svec_new[3:])


class RK4Sim:
    def __init__(self, h, cbody_list, sc_list, t_start, t_end, title, center="EARTH"):
        self.h = h
        self.cbody_list = cbody_list
        self.sc_list = sc_list
        self.t_start = t_start
        self.t_end = t_end
        self.center = center
        self.title = title
        self.hcount = int((self.t_end - self.t_start) / h)
        self.pos_arr = np.zeros((self.hcount, 3))

        earth_state_sun = earth.get_svec(t_start, frame="SUN")
        self.sc_list[0].svec = self.sc_list[0].svec + earth_state_sun


    def tstep(self, sc_id, et):
        scraft = self.sc_list[sc_id]

        k1_v = self.h * scraft.svec[3:]
        k1_a = self.h * scraft.f_gravity(self.cbody_list, et) / scraft.m_wet

        svec_temp2 = np.zeros(6)
        svec_temp2[:3] = scraft.svec[:3] + k1_v / 2
        svec_temp2[3:] = scraft.svec[3:] + k1_a / 2
        scraft.update_temp(self.h / 2, svec_temp2)

        k2_v = self.h * scraft.svec_temp[3:]
        k2_a = self.h * scraft.f_gravity(self.cbody_list, et + (self.h / 2), temp=True) / scraft.m_wet_temp

        svec_temp3 = np.zeros(6)
        svec_temp3[:3] = scraft.svec[:3] + k2_v / 2
        svec_temp3[3:] = scraft.svec[3:] + k2_a / 2
        scraft.update_temp(self.h / 2, svec_temp3)

        k3_v = self.h * scraft.svec_temp[3:]
        k3_a = self.h * scraft.f_gravity(self.cbody_list, et + (self.h / 2), temp=True) / scraft.m_wet_temp

        svec_temp4 = np.zeros(6)
        svec_temp4[:3] = scraft.svec[:3] + k3_v
        svec_temp4[3:] = scraft.svec[3:] + k3_a
        scraft.update_temp(self.h / 2, svec_temp4)

        k4_v = self.h * scraft.svec_temp[3:]
        k4_a = self.h * scraft.f_gravity(self.cbody_list, et + self.h, temp=True) / scraft.m_wet_temp

        svec_new = np.zeros(6)
        svec_new[:3] = scraft.svec[:3] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        svec_new[3:] = scraft.svec[3:] + (k1_a + 2 * k2_a + 2 * k3_a + k4_a) / 6

        scraft.update(self.h, svec_new)
        return svec_new[:3]

    def run(self):
        et = self.t_start
        for i in tqdm(range(self.hcount)):
            self.pos_arr[i, :] = self.tstep(0, et)
            et += self.h

    

    

    def visualize(self, animate_rate=1):
        stride = 500  # integration-step sampling stride
        # --- Spacecraft trajectory data (sampled) ---
        sat_x = self.pos_arr[::stride, 0]
        sat_y = self.pos_arr[::stride, 1]
        sat_z = self.pos_arr[::stride, 2]

        # --- Create figure and 3D axis ---
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.grid(False)
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.set_zlabel("Z (m)", color="white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.tick_params(axis="z", colors="white")
        ax.xaxis.line.set_color("gray")
        ax.yaxis.line.set_color("gray")
        ax.zaxis.line.set_color("gray")
        ax.xaxis.line.set_linewidth(2)
        ax.yaxis.line.set_linewidth(2)
        ax.zaxis.line.set_linewidth(2)
        fig.suptitle(self.title, color="white", fontsize=16)

        # --- Precompute a unit sphere mesh ---
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_unit = np.outer(np.cos(u), np.sin(v))
        y_unit = np.outer(np.sin(u), np.sin(v))
        z_unit = np.outer(np.ones(np.size(u)), np.cos(v))

        # Dictionaries to hold the artists for each celestial body (keyed by index)
        body_surfaces = {}
        body_labels = {}
        body_centers = {}  # for the x-shaped scatter markers

        # --- Prepare spacecraft trajectory lines ---
        line_old, = ax.plot([], [], [], label="Orbit Trajectory", color="red", lw=1)
        line_recent, = ax.plot([], [], [], color="white", lw=3)

        # Live text annotations.
        time_text = fig.text(0.05, 0.925, "Time:", color="white", fontsize=12)
        et_text = fig.text(0.05, 0.900, "", color="white", fontsize=10)
        met_text = fig.text(0.05, 0.875, "", color="white", fontsize=10)
        state_vec_text = fig.text(0.05, 0.8250, "State Vector: (ECI)", color="white", fontsize=12)
        pos_text = fig.text(0.05, 0.800, "", color="white", fontsize=10)
        vel_text = fig.text(0.05, 0.775, "", color="white", fontsize=10)

        # Mutable container to store the current frame.
        current_frame = [0]

        # Bounding box half-size (zoom level)
        bb_half = [2e8]  # Initial half-width; adjust as needed

        # --- Scroll event handler to adjust zoom ---
        def on_scroll(event):
            if event.button == "up":
                bb_half[0] *= 0.9
            elif event.button == "down":
                bb_half[0] *= 1.1
            update(current_frame[0])
        fig.canvas.mpl_connect("scroll_event", on_scroll)

        # Update function for the animation.
        def update(num):
            ax.set_title(self.title)
            current_frame[0] = num
            # Update the trajectory lines.
            if num == 0:
                pass
            elif num >= 2:
                line_old.set_data(sat_x[:num - 1], sat_y[:num - 1])
                line_old.set_3d_properties(sat_z[:num - 1])
                line_recent.set_data(sat_x[num - 2:num], sat_y[num - 2:num])
                line_recent.set_3d_properties(sat_z[num - 2:num])
            else:
                line_old.set_data(sat_x[:num], sat_y[:num])
                line_old.set_3d_properties(sat_z[:num])
                line_recent.set_data([], [])
                line_recent.set_3d_properties([])

            # Compute current simulation time.
            current_et = self.t_start + num * stride * self.h

            # --- Update each celestial body's position, surface, center marker, and name label ---
            for idx, body in enumerate(self.cbody_list):
                # Retrieve current position.
                pos = body.get_svec(current_et)[:3]
                # Compute the scaled sphere coordinates for the body.
                x_body = body.radius * x_unit + pos[0]
                y_body = body.radius * y_unit + pos[1]
                z_body = body.radius * z_unit + pos[2]
                # Remove any existing surface for this body.
                if idx in body_surfaces and body_surfaces[idx] is not None:
                    body_surfaces[idx].remove()
                # Plot the celestial body using its color property.
                body_surfaces[idx] = ax.plot_surface(
                    x_body, y_body, z_body,
                    color=body.color, alpha=0.5, rstride=4, cstride=4
                )

                # Remove any existing center marker and add a new one.
                if idx in body_centers and body_centers[idx] is not None:
                    body_centers[idx].remove()
                # The scatter marker uses an 'x' shape; its size (s=50) can be adjusted.
                body_centers[idx] = ax.scatter(
                    pos[0], pos[1], pos[2], marker="x", color=body.color, alpha=0.5, s=20
                )

                # Remove any existing text label and add a new one.
                if idx in body_labels and body_labels[idx] is not None:
                    body_labels[idx].remove()
                # Place the text label slightly above the body.
                label_x = pos[0]
                label_y = pos[1]
                label_z = pos[2] + body.radius * 1.05  # Offset; adjust if needed.
                body_labels[idx] = ax.text(
                    label_x, label_y, label_z, body.name,
                    color="white", fontsize=9, ha="center", va="bottom"
                )

            # Get spacecraft's current position.
            idx_sc = min(num, len(sat_x) - 1)
            current_sc_x = sat_x[idx_sc]
            current_sc_y = sat_y[idx_sc]
            current_sc_z = sat_z[idx_sc]

            # Set axis limits centered on the spacecraft.
            ax.set_xlim(current_sc_x - bb_half[0], current_sc_x + bb_half[0])
            ax.set_ylim(current_sc_y - bb_half[0], current_sc_y + bb_half[0])
            ax.set_zlim(current_sc_z - bb_half[0], current_sc_z + bb_half[0])

            # Update live annotations.
            et_text.set_text(f"Current J2000 ET: {current_et:.2f} s")
            met = current_et - self.t_start
            met_text.set_text(f"Mission Elapsed Time: {met:.2f} s")
            pos_text.set_text(
                f"Spacecraft: ({(current_sc_x/1000):.2f}, {(current_sc_y/1000):.2f}, {(current_sc_z/1000):.2f}) km"
            )

            # Return updated artists.
            return (line_old, line_recent, 
                    *list(body_surfaces.values()), 
                    *list(body_labels.values()), 
                    *list(body_centers.values()), 
                    et_text)

        # Create the animation.
        ani = animation.FuncAnimation(
            fig, update, frames=len(sat_x),
            interval=animate_rate, blit=False, repeat=False
        )

        # --- Add interactive buttons for simulation control ---
        ax_stop = fig.add_axes([0.8, 0.9, 0.08, 0.035])
        btn_stop = Button(ax_stop, "Stop", color="lightgray", hovercolor="gray")
        def stop_animation(event):
            ani.event_source.stop()
        btn_stop.on_clicked(stop_animation)

        ax_start = fig.add_axes([0.8, 0.83, 0.08, 0.035])
        btn_start = Button(ax_start, "Start", color="lightgray", hovercolor="gray")
        def start_animation(event):
            nonlocal ani
            ani.event_source.start()
        btn_start.on_clicked(start_animation)

        ax_restart = fig.add_axes([0.8, 0.76, 0.08, 0.035])
        btn_restart = Button(ax_restart, "Restart", color="lightgray", hovercolor="gray")
        def restart_animation(event):
            nonlocal ani
            if ani.event_source is not None:
                ani.event_source.stop()
            ani = animation.FuncAnimation(
                fig, update, frames=len(sat_x),
                interval=animate_rate, blit=False, repeat=False
            )
        btn_restart.on_clicked(restart_animation)

        plt.show()





# Define Earth, Moon, and spacecraft, then execute the simulation
earth = CBody("Earth", "EARTH", 5.9722e24, 6378137, color="blue")
luna = CBody("Moon", "MOON", 7.34767309e22, 1737400, color="white")
sol  = CBody("Sun", "10", 1.9891e30, 6.957e8, color="yellow")
# Sample spacecraft state vector: [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z]
sat1 = SCraft(100, 80, 1000, 320, np.array([4404364.154715429, -4311452.937854692, -3333241.5900105746, 8626.193135065847, 5867.596093273166, -1661.187545094598]))
sim1 = RK4Sim(10, [earth, luna, sol], [sat1], 298635469, 298635469+14*36000, "Lunar Reconnaissance Orbiter Lunar Injection and Transfer")

sim1.run()
sim1.visualize()
