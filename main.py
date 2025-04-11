import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import spiceypy as spice
from spiceypy.utils.support_types import SPICEDOUBLE_CELL
from tqdm import tqdm
import pickle


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
        self.simframe = "EARTH"

    def f_gravity(self, cbody_list, et, temp=False):
        f_g_sum = 0
        if not temp:
            m_use = self.m_wet
            pos_sc = self.svec[:3]
        else:
            m_use = self.m_wet_temp
            pos_sc = self.svec_temp[:3]

        for cbody in cbody_list:
            rvec = pos_sc - cbody.get_svec(et, frame=self.simframe)[:3]
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
    def __init__(self, h, scraft, t_start, t_end, title, system):
        self.h = h
        self.scraft = scraft
        self.t_start = t_start
        self.t_end = t_end
        self.title = title
        self.system = system
        self.hcount = int((self.t_end - self.t_start) / h)
        self.pos_arr = np.zeros((self.hcount, 4))
        

        self.cbody_list = []

        Sun     = CBody("Sun", "10", 1.9891e30, 6.957e8, color="yellow")
        Mercury = CBody("Mercury", "MERCURY", 3.3011e23, 2439700, color="gray")
        Venus   = CBody("Venus", "VENUS", 4.8675e24, 6051800, color="palegoldenrod")
        Earth   = CBody("Earth", "EARTH", 5.9722e24, 6378137, color="blue")
        Moon    = CBody("Moon", "MOON", 7.34767309e22, 1737400, color="white")
        Mars    = CBody("Mars", "MARS BARYCENTER", 6.4171e23, 3389500, color="red")
        Jupiter = CBody("Jupiter", "JUPITER BARYCENTER", 1.89813e27, 69911000, color="orange")
        Saturn  = CBody("Saturn", "SATURN BARYCENTER", 5.6834e26, 58232000, color="gold") 
        Uranus  = CBody("Uranus", "URANUS BARYCENTER", 8.6810e25, 25362000, color="cyan")
        Neptune = CBody("Neptune", "NEPTUNE BARYCENTER", 1.02413e26, 24622000, color="navy")

        if self.system == "EM":
            self.cbody_list = [Earth, Moon]
            self.simframe = "EARTH"
        elif self.system == "FS":
            self.cbody_list = [Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn, Uranus, Neptune]
            earth_state_sun = Earth.get_svec(t_start, frame="SUN")
            self.scraft.svec = self.scraft.svec + earth_state_sun
            self.simframe = "SUN"
            self.scraft.simframe = "SUN"


    def tstep(self, et):
        scraft = self.scraft

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
            self.pos_arr[i, :3] = self.tstep(et)
            self.pos_arr[i, 3] = et
            et += self.h
    


class FData:
    def __init__(self, simdata):
        self.pos_arr    = simdata.pos_arr
        self.title      = simdata.title
        self.system     = simdata.system
        self.h          = simdata.h
        self.cbody_list = simdata.cbody_list
        self.simframe   = simdata.simframe

        self.t_start = self.pos_arr[0, 3]
        self.t_end   = self.pos_arr[-1, 3]


        print("Title Stored in FData Class:", self.title)
        print(self.t_start)
        print(self.t_end)

    
    def save_to_file(self, filename):
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"FData object saved successfully to {filename}")
        except Exception as e:
            print(f"Error saving FData object: {e}")

    @classmethod
    def load_from_file(cls, filename):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            print(f"FData object loaded successfully from {filename}")
            return data
        except Exception as e:
            print(f"Error loading FData object: {e}")
            return None
        

    def compare(self, flight_ephem, scraft_id, n_points):
        # Load necessary kernels.
        spice.furnsh(flight_ephem)
        spice.furnsh("ephems/de440.bsp")
        spice.furnsh("naif0012.tls")

        # Get the coverage intervals for the flight ephemeris.
        coverage_window = SPICEDOUBLE_CELL(1000)
        spice.spkcov(flight_ephem, scraft_id, coverage_window)
        n_intervals = spice.wncard(coverage_window)
        print(f"Found {n_intervals} coverage interval(s) for object id {scraft_id}.")

        # Print each coverage interval.
        for i in range(n_intervals):
            (start_et, stop_et) = spice.wnfetd(coverage_window, i)
            print(f"Interval {i+1}: Start ET = {start_et}, Stop ET = {stop_et}")
            start_utc = spice.et2utc(start_et, 'C', 3)
            stop_utc  = spice.et2utc(stop_et, 'C', 3)
            print(f"           Start UTC: {start_utc}, Stop UTC: {stop_utc}")
        print("")

        # Prepare an array to store the true state positions.
        self.true_pos_arr = np.zeros_like(self.pos_arr)

        # Loop over all simulation times.
        for i in range(self.pos_arr.shape[0]):
            et = self.pos_arr[i, 3]
            # Retrieve the state vector from SPICE (note: target is passed as a string).
            state, light_time = spice.spkezr(str(scraft_id), et, "J2000", "NONE", self.simframe)
            # The simulation uses meters so convert state from km to m.
            self.true_pos_arr[i, :3] = state[:3] * 1000
            self.true_pos_arr[i, 3] = et

        # Compute the error norm between your simulation and the SPICE state.
        error_norm = np.linalg.norm(self.pos_arr[:, :3] - self.true_pos_arr[:, :3], axis=1)
        time_arr = self.pos_arr[:, 3]

        # Compute summary statistics.
        max_error   = np.max(error_norm)
        mean_error  = np.mean(error_norm)
        final_error = error_norm[-1]

        print(f"Max error:   {max_error:.6f}")
        print(f"Mean error:  {mean_error:.6f}")
        print(f"Final error: {final_error:.6f}")

        # Sample equally spaced indices for the graph
        total_points = len(time_arr)
        sample_indices = np.linspace(0, total_points - 1, n_points, dtype=int)
        sampled_time   = time_arr[sample_indices]
        sampled_error  = error_norm[sample_indices]

        # Plot error vs time for the sampled points.
        plt.figure(figsize=(10, 6))
        plt.plot(sampled_time, sampled_error)
        plt.title("RK4 Position Error Over Time (Filtered)")
        plt.xlabel("Time [s]")
        plt.ylabel("Position Error [m]")
        plt.grid(True)
        plt.show()

        # Unload kernels.
        spice.unload(flight_ephem)
        spice.unload("ephems/de440.bsp")
        spice.unload("naif0012.tls")





    def visualize(self, time_warp=100):
        # Fix the animate rate to 1 ms (or 1 unit as used in interval)
        animate_rate = 1

        # Compute the frame times from simulation start to end using the given time warp factor.
        frame_times = np.arange(self.t_start, self.t_end, time_warp)
        # Find the corresponding indices in the simulation data.
        indices = np.searchsorted(self.pos_arr[:, 3], frame_times)
        # Ensure we don't go out-of-bounds.
        indices = np.clip(indices, 0, len(self.pos_arr) - 1)

        # Precompute spacecraft trajectory data (sampled).
        sat_x = self.pos_arr[indices, 0]
        sat_y = self.pos_arr[indices, 1]
        sat_z = self.pos_arr[indices, 2]

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

            # Get current frame time from simulation data.
            current_index = indices[num]
            current_et = self.pos_arr[current_index, 3]

            # Get spacecraft's current position.
            current_sc_x = sat_x[num]
            current_sc_y = sat_y[num]
            current_sc_z = sat_z[num]
            current_sc_pos = np.array([current_sc_x, current_sc_y, current_sc_z])

            # --- Update each celestial body's position and visualization ---
            for idx, body in enumerate(self.cbody_list):
                pos = body.get_svec(current_et, frame=self.simframe)[:3]
                # Calculate distance from the spacecraft to the body's center.
                distance = np.linalg.norm(pos - current_sc_pos)

                # Check if distance is below the threshold (1e9 m).
                if distance < 1e9:
                    # Compute sphere coordinates only if within threshold.
                    x_body = body.radius * x_unit + pos[0]
                    y_body = body.radius * y_unit + pos[1]
                    z_body = body.radius * z_unit + pos[2]
                    # Remove any existing surface and then replot.
                    if idx in body_surfaces and body_surfaces[idx] is not None:
                        body_surfaces[idx].remove()
                    body_surfaces[idx] = ax.plot_surface(
                        x_body, y_body, z_body,
                        color=body.color, alpha=0.5, rstride=4, cstride=4
                    )
                else:
                    # If beyond the threshold, ensure sphere is removed.
                    if idx in body_surfaces and body_surfaces[idx] is not None:
                        body_surfaces[idx].remove()
                        body_surfaces[idx] = None

                # Update the center marker ("x").
                if idx in body_centers and body_centers[idx] is not None:
                    body_centers[idx].remove()
                body_centers[idx] = ax.scatter(
                    pos[0], pos[1], pos[2], marker="x", color=body.color, alpha=0.5, s=20
                )

                # Update the text label.
                if idx in body_labels and body_labels[idx] is not None:
                    body_labels[idx].remove()
                label_x = pos[0]
                label_y = pos[1]
                label_z = pos[2] + body.radius * 1.05
                body_labels[idx] = ax.text(
                    label_x, label_y, label_z, body.name,
                    color="white", fontsize=9, ha="center", va="bottom"
                )

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


def compare_fdata_list(fdata_list, flight_ephem, scraft_id, n_points):

    # --- Load the necessary SPICE kernels ---
    spice.furnsh(flight_ephem)
    spice.furnsh("ephems/de440.bsp")
    spice.furnsh("naif0012.tls")
    
    # --- Retrieve and print flight ephemeris coverage intervals ---
    coverage_window = SPICEDOUBLE_CELL(1000)
    spice.spkcov(flight_ephem, scraft_id, coverage_window)
    n_intervals = spice.wncard(coverage_window)
    print(f"Found {n_intervals} coverage interval(s) for object id {scraft_id} in flight ephemeris '{flight_ephem}':")
    for i in range(n_intervals):
        (start_et, stop_et) = spice.wnfetd(coverage_window, i)
        start_utc = spice.et2utc(start_et, 'C', 3)
        stop_utc  = spice.et2utc(stop_et, 'C', 3)
        print(f"  Interval {i+1}: Start ET = {start_et}, Stop ET = {stop_et}")
        print(f"              Start UTC: {start_utc}, Stop UTC: {stop_utc}")
    print("")

    # --- Create a new plot ---
    plt.figure(figsize=(10, 6))
    
    # --- Process each FData simulation ---
    for fdata in fdata_list:
        # Create an array to store true state positions (using same shape as the simulation positions)
        true_pos_arr = np.zeros_like(fdata.pos_arr)

        # Loop over all simulation times to retrieve true spacecraft state from SPICE.
        for i in range(fdata.pos_arr.shape[0]):
            et = fdata.pos_arr[i, 3]
            state, light_time = spice.spkezr(str(scraft_id), et, "J2000", "NONE", fdata.simframe)
            # Convert from km (SPICE default) to m and store into true position array.
            true_pos_arr[i, :3] = state[:3] * 1000
            true_pos_arr[i, 3] = et

        # Compute the error norm (L2 norm) between simulated and SPICE true positions.
        error_norm = np.linalg.norm(fdata.pos_arr[:, :3] - true_pos_arr[:, :3], axis=1)
        time_arr = fdata.pos_arr[:, 3]
        
        # Compute and print summary error statistics.
        max_error = np.max(error_norm)
        mean_error = np.mean(error_norm)
        final_error = error_norm[-1]
        print(f"Simulation '{fdata.title}':")
        print(f"  Max error:   {max_error:.6f} m")
        print(f"  Mean error:  {mean_error:.6f} m")
        print(f"  Final error: {final_error:.6f} m\n")
        
        # Sample the data to plot only n_points for clarity.
        total_points = len(time_arr)
        sample_indices = np.linspace(0, total_points - 1, n_points, dtype=int)
        sampled_time = time_arr[sample_indices]
        sampled_error = error_norm[sample_indices]
        
        # Plot the error vs. time for this simulation.
        plt.plot(sampled_time, sampled_error, label=fdata.title)

    # --- Finalize the plot ---
    plt.title("RK4 Position Error Over Time (All Simulations)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Unload SPICE kernels ---
    spice.unload(flight_ephem)
    spice.unload("ephems/de440.bsp")
    spice.unload("naif0012.tls")




# Sample spacecraft state vector: [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z]
#sat1 = SCraft(100, 80, 1000, 320, np.array([4404364.154715429, -4311452.937854692, -3333241.5900105746, 8626.193135065847, 5867.596093273166, -1661.187545094598]))
sat1 = SCraft(100, 80, 1000, 320, np.array([7447942.501507646, -1408650.136251841, 1985206.7413388218, 5987.824140223813, 8866.625464085342, 9480.132017400176]))
# h, scraft, t_start, t_end, title, system
#sim1 = RK4Sim(1, sat1, 298635469, 298635469+14*36000, "Lunar Reconnaissance Orbiter Lunar Injection and Transfer", "EM")

sim1 = RK4Sim(20, sat1, -705788798.8171841, -705788798.8171841+4.5*31557600, "Voyager 2", "FS")

sim1.run()

fdata1 = FData(sim1)

fdata1.save_to_file("fdata_v2_fs.pkl")

#fdata1 = FData.load_from_file("fdata_test3_fs.pkl")

#fdata1.visualize(time_warp = 1000)
#fdata1.compare("flight_ephems/lrorg_2009169_2010001_v01.bsp", -85, 10000)
#fdata1.compare("flight_ephems/Voyager_2.m05016u.merged.bsp", -32)
