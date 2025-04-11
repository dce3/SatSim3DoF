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
spice.furnsh("ephems/naif0012.tls")  # Handles leap seconds


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
        # Changed from 4 columns to 7 columns: [x, y, z, vx, vy, vz, et]
        self.pos_arr = np.zeros((self.hcount, 7))
        
        self.cbody_list = []

        Sun     = CBody("Sun", "SUN", 1.9891e30, 6.957e8, color="yellow")
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
        return svec_new  # returns full state vector [x, y, z, vx, vy, vz]

    def run(self):
        et = self.t_start
        for i in tqdm(range(self.hcount)):
            svec_new = self.tstep(et)  # svec_new is [x, y, z, vx, vy, vz]
            self.pos_arr[i, :6] = svec_new   # store full state vector
            self.pos_arr[i, 6] = et          # store ephemeris time
            et += self.h

    


class FData:
    def __init__(self, simdata):
        self.pos_arr    = simdata.pos_arr
        self.title      = simdata.title
        self.system     = simdata.system
        self.h          = simdata.h
        self.cbody_list = simdata.cbody_list
        self.simframe   = simdata.simframe

        self.t_start = self.pos_arr[0, 6]
        self.t_end   = self.pos_arr[-1, 6]


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
        spice.furnsh("ephems/naif0012.tls")

        # Get the coverage intervals for the flight ephemeris.
        coverage_window = SPICEDOUBLE_CELL(1000)
        spice.spkcov(flight_ephem, scraft_id, coverage_window)
        n_intervals = spice.wncard(coverage_window)
        print(f"Found {n_intervals} coverage interval(s) for object id {scraft_id}.")

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
            et = self.pos_arr[i, 6]  # Updated index for time
            state, light_time = spice.spkezr(str(scraft_id), et, "J2000", "NONE", self.simframe)
            self.true_pos_arr[i, :3] = state[:3] * 1000
            self.true_pos_arr[i, 6] = et

        error_norm = np.linalg.norm(self.pos_arr[:, :3] - self.true_pos_arr[:, :3], axis=1)
        time_arr = self.pos_arr[:, 6]

        max_error   = np.max(error_norm)
        mean_error  = np.mean(error_norm)
        final_error = error_norm[-1]

        print(f"Max error:   {max_error:.6f}")
        print(f"Mean error:  {mean_error:.6f}")
        print(f"Final error: {final_error:.6f}")

        total_points = len(time_arr)
        sample_indices = np.linspace(0, total_points - 1, n_points, dtype=int)
        sampled_time   = time_arr[sample_indices]
        sampled_error  = error_norm[sample_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(sampled_time, sampled_error)
        plt.title("RK4 Position Error Over Time (Filtered)")
        plt.xlabel("Time [s]")
        plt.ylabel("Position Error [m]")
        plt.grid(True)
        plt.show()

        spice.unload(flight_ephem)
        spice.unload("ephems/de440.bsp")
        spice.unload("ephems/naif0012.tls")


    def compare_velocity(self, flight_ephem, scraft_id, n_points, end_et=None):
        # Load necessary kernels.
        spice.furnsh(flight_ephem)
        spice.furnsh("ephems/de440.bsp")
        spice.furnsh("ephems/naif0012.tls")
        
        # Get the coverage intervals for the flight ephemeris.
        coverage_window = SPICEDOUBLE_CELL(1000)
        spice.spkcov(flight_ephem, scraft_id, coverage_window)
        n_intervals = spice.wncard(coverage_window)
        print(f"Found {n_intervals} coverage interval(s) for object id {scraft_id}.")
        for i in range(n_intervals):
            (start_et, stop_et) = spice.wnfetd(coverage_window, i)
            print(f"Interval {i+1}: Start ET = {start_et}, Stop ET = {stop_et}")
            start_utc = spice.et2utc(start_et, 'C', 3)
            stop_utc  = spice.et2utc(stop_et, 'C', 3)
            print(f"           Start UTC: {start_utc}, Stop UTC: {stop_utc}")
        print("")

        # Optionally filter the simulation data if an end_et is provided.
        if end_et is not None:
            mask = self.pos_arr[:, 6] <= end_et
            if not np.any(mask):
                print(f"Simulation '{self.title}' has no data up to the specified end ET {end_et}.")
                spice.unload(flight_ephem)
                spice.unload("ephems/de440.bsp")
                spice.unload("ephems/naif0012.tls")
                return
            pos_arr_subset = self.pos_arr[mask]
        else:
            pos_arr_subset = self.pos_arr

        # Prepare an array to store the true velocity state for the subset.
        true_state_arr = np.zeros_like(pos_arr_subset)
        for i in range(pos_arr_subset.shape[0]):
            et = pos_arr_subset[i, 6]  # Time is stored in column 6.
            state, light_time = spice.spkezr(str(scraft_id), et, "J2000", "NONE", self.simframe)
            # Convert SPICE velocity from km/s to m/s and store in columns 3 to 5.
            true_state_arr[i, 3:6] = state[3:] * 1000
            true_state_arr[i, 6] = et

        # Calculate the velocity error norm (difference between simulated and true velocities).
        velocity_error = np.linalg.norm(pos_arr_subset[:, 3:6] - true_state_arr[:, 3:6], axis=1)
        time_arr = pos_arr_subset[:, 6]

        max_error   = np.max(velocity_error)
        mean_error  = np.mean(velocity_error)
        final_error = velocity_error[-1]

        print(f"Max velocity error:   {max_error:.6f} m/s")
        print(f"Mean velocity error:  {mean_error:.6f} m/s")
        print(f"Final velocity error: {final_error:.6f} m/s")

        total_points = len(time_arr)
        sample_indices = np.linspace(0, total_points - 1, n_points, dtype=int)
        sampled_time   = time_arr[sample_indices]
        sampled_error  = velocity_error[sample_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(sampled_time, sampled_error)
        plt.xlabel("J2000 Ephemeris Time [s]")
        plt.ylabel("Velocity Error [m/s]")
        plt.grid(True)
        plt.show()

        spice.unload(flight_ephem)
        spice.unload("ephems/de440.bsp")
        spice.unload("ephems/naif0012.tls")

    def compare_acceleration(self, flight_ephem, scraft_id, n_points, end_et=None):
        # Load necessary kernels.
        spice.furnsh(flight_ephem)
        spice.furnsh("ephems/de440.bsp")
        spice.furnsh("ephems/naif0012.tls")
        
        # Get the coverage intervals for the flight ephemeris.
        coverage_window = SPICEDOUBLE_CELL(1000)
        spice.spkcov(flight_ephem, scraft_id, coverage_window)
        n_intervals = spice.wncard(coverage_window)
        print(f"Found {n_intervals} coverage interval(s) for object id {scraft_id}.")
        for i in range(n_intervals):
            (start_et_int, stop_et_int) = spice.wnfetd(coverage_window, i)
            print(f"Interval {i+1}: Start ET = {start_et_int}, Stop ET = {stop_et_int}")
            start_utc = spice.et2utc(start_et_int, 'C', 3)
            stop_utc  = spice.et2utc(stop_et_int, 'C', 3)
            print(f"           Start UTC: {start_utc}, Stop UTC: {stop_utc}")
        print("")
        
        # Optionally filter the simulation data if an end_et is provided.
        if end_et is not None:
            mask = self.pos_arr[:, 6] <= end_et
            if not np.any(mask):
                print(f"Simulation '{self.title}' has no data up to the specified end ET {end_et}.")
                spice.unload(flight_ephem)
                spice.unload("ephems/de440.bsp")
                spice.unload("ephems/naif0012.tls")
                return
            pos_arr_subset = self.pos_arr[mask]
        else:
            pos_arr_subset = self.pos_arr
        
        # Prepare an array to store the true velocity state for the subset.
        true_state_arr = np.zeros_like(pos_arr_subset)
        for i in range(pos_arr_subset.shape[0]):
            et = pos_arr_subset[i, 6]  # Time is stored in column 6.
            state, light_time = spice.spkezr(str(scraft_id), et, "J2000", "NONE", self.simframe)
            # Convert SPICE velocity from km/s to m/s and store in columns 3 to 5.
            true_state_arr[i, 3:6] = state[3:] * 1000
            true_state_arr[i, 6] = et
        
        # Compute the acceleration (using central finite differences) for both the simulation and SPICE-derived data.
        # pos_arr_subset[:, 3:6] are the simulated velocities.
        # Use the corresponding times in column 6 to compute the gradients.
        sim_acc = np.gradient(pos_arr_subset[:, 3:6], pos_arr_subset[:, 6], axis=0)
        true_acc = np.gradient(true_state_arr[:, 3:6], true_state_arr[:, 6], axis=0)
        
        # Compute the acceleration error as the L2 norm difference between the simulated and true accelerations.
        acceleration_error = np.linalg.norm(sim_acc - true_acc, axis=1)
        time_arr = pos_arr_subset[:, 6]
        
        max_error   = np.max(acceleration_error)
        mean_error  = np.mean(acceleration_error)
        final_error = acceleration_error[-1]
        
        print(f"Max acceleration error:   {max_error:.6f} m/s²")
        print(f"Mean acceleration error:  {mean_error:.6f} m/s²")
        print(f"Final acceleration error: {final_error:.6f} m/s²")
        
        total_points = len(time_arr)
        sample_indices = np.linspace(0, total_points - 1, n_points, dtype=int)
        sampled_time   = time_arr[sample_indices]
        sampled_error  = acceleration_error[sample_indices]
        
        # Plot the acceleration error vs. time.
        plt.figure(figsize=(10, 6))
        plt.plot(sampled_time, sampled_error, lw=2)
        plt.xlabel("J2000 Ephemeris Time [s]", fontsize=12)
        plt.ylabel("Acceleration Error [m/s²]", fontsize=12)
        plt.title("RK4 Acceleration Error Over Time (Filtered)", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Unload kernels.
        spice.unload(flight_ephem)
        spice.unload("ephems/de440.bsp")
        spice.unload("ephems/naif0012.tls")


    def compare_jerk(self, flight_ephem, scraft_id, n_points, end_et=None):
        # Load necessary kernels.
        spice.furnsh(flight_ephem)
        spice.furnsh("ephems/de440.bsp")
        spice.furnsh("ephems/naif0012.tls")
        
        # Get the coverage intervals for the flight ephemeris.
        coverage_window = SPICEDOUBLE_CELL(1000)
        spice.spkcov(flight_ephem, scraft_id, coverage_window)
        n_intervals = spice.wncard(coverage_window)
        print(f"Found {n_intervals} coverage interval(s) for object id {scraft_id}.")
        for i in range(n_intervals):
            (start_et_int, stop_et_int) = spice.wnfetd(coverage_window, i)
            print(f"Interval {i+1}: Start ET = {start_et_int}, Stop ET = {stop_et_int}")
            start_utc = spice.et2utc(start_et_int, 'C', 3)
            stop_utc  = spice.et2utc(stop_et_int, 'C', 3)
            print(f"           Start UTC: {start_utc}, Stop UTC: {stop_utc}")
        print("")
        
        # Optionally filter the simulation data if an end_et is provided.
        if end_et is not None:
            mask = self.pos_arr[:, 6] <= end_et
            if not np.any(mask):
                print(f"Simulation '{self.title}' has no data up to the specified end ET {end_et}.")
                spice.unload(flight_ephem)
                spice.unload("ephems/de440.bsp")
                spice.unload("ephems/naif0012.tls")
                return
            pos_arr_subset = self.pos_arr[mask]
        else:
            pos_arr_subset = self.pos_arr
        
        # Prepare an array to store the true velocity state for the subset.
        true_state_arr = np.zeros_like(pos_arr_subset)
        for i in range(pos_arr_subset.shape[0]):
            et = pos_arr_subset[i, 6]  # Time is stored in column 6.
            state, light_time = spice.spkezr(str(scraft_id), et, "J2000", "NONE", self.simframe)
            # Convert SPICE velocity from km/s to m/s and store in columns 3 to 5.
            true_state_arr[i, 3:6] = state[3:] * 1000
            true_state_arr[i, 6] = et
        
        # Compute acceleration (first derivative) using central differences.
        sim_acc = np.gradient(pos_arr_subset[:, 3:6], pos_arr_subset[:, 6], axis=0)
        true_acc = np.gradient(true_state_arr[:, 3:6], true_state_arr[:, 6], axis=0)
        
        # Compute jerk (second derivative, i.e. derivative of acceleration) using central differences.
        sim_jerk = np.gradient(sim_acc, pos_arr_subset[:, 6], axis=0)
        true_jerk = np.gradient(true_acc, true_state_arr[:, 6], axis=0)
        
        # Compute the jerk error as the L2 norm difference between simulated and true jerk.
        jerk_error = np.linalg.norm(sim_jerk - true_jerk, axis=1)
        time_arr = pos_arr_subset[:, 6]
        
        max_error   = np.max(jerk_error)
        mean_error  = np.mean(jerk_error)
        final_error = jerk_error[-1]
        
        print(f"Max jerk error:   {max_error:.6f} m/s³")
        print(f"Mean jerk error:  {mean_error:.6f} m/s³")
        print(f"Final jerk error: {final_error:.6f} m/s³")
        
        total_points = len(time_arr)
        sample_indices = np.linspace(0, total_points - 1, n_points, dtype=int)
        sampled_time   = time_arr[sample_indices]
        sampled_error  = jerk_error[sample_indices]
        
        # Plot the jerk error vs. time.
        plt.figure(figsize=(10, 6))
        plt.plot(sampled_time, sampled_error, lw=2)
        plt.xlabel("J2000 Ephemeris Time [s]", fontsize=12)
        plt.ylabel("Jerk Error [m/s³]", fontsize=12)
        plt.title("RK4 Jerk Error Over Time (Filtered)", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Unload kernels.
        spice.unload(flight_ephem)
        spice.unload("ephems/de440.bsp")
        spice.unload("ephems/naif0012.tls")


    def visualize(self, time_warp=1000):
        animate_rate = 1
        frame_times = np.arange(self.t_start, self.t_end, time_warp)
        indices = np.searchsorted(self.pos_arr[:, 6], frame_times)
        indices = np.clip(indices, 0, len(self.pos_arr) - 1)

        sat_x = self.pos_arr[indices, 0]
        sat_y = self.pos_arr[indices, 1]
        sat_z = self.pos_arr[indices, 2]

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

        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_unit = np.outer(np.cos(u), np.sin(v))
        y_unit = np.outer(np.sin(u), np.sin(v))
        z_unit = np.outer(np.ones(np.size(u)), np.cos(v))

        body_surfaces = {}
        body_labels = {}
        body_centers = {}

        line_old, = ax.plot([], [], [], label="Orbit Trajectory", color="red", lw=1)
        line_recent, = ax.plot([], [], [], color="white", lw=3)

        time_text = fig.text(0.05, 0.925, "Time:", color="white", fontsize=12)
        et_text = fig.text(0.05, 0.900, "", color="white", fontsize=10)
        met_text = fig.text(0.05, 0.875, "", color="white", fontsize=10)
        state_vec_text = fig.text(0.05, 0.8250, "State Vector: (ECI/SCI)", color="white", fontsize=12)
        pos_text = fig.text(0.05, 0.800, "", color="white", fontsize=10)
        vel_text = fig.text(0.05, 0.775, "", color="white", fontsize=10)

        current_frame = [0]
        bb_half = [2e8]

        def on_scroll(event):
            if event.button == "up":
                bb_half[0] *= 0.9
            elif event.button == "down":
                bb_half[0] *= 1.1
            update(current_frame[0])
        fig.canvas.mpl_connect("scroll_event", on_scroll)

        def update(num):
            ax.set_title(self.title)
            current_frame[0] = num
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

            current_index = indices[num]
            # Updated: using column 6 for ephemeris time
            current_et = self.pos_arr[current_index, 6]

            current_sc_x = sat_x[num]
            current_sc_y = sat_y[num]
            current_sc_z = sat_z[num]
            current_sc_pos = np.array([current_sc_x, current_sc_y, current_sc_z])

            for idx, body in enumerate(self.cbody_list):
                pos = body.get_svec(current_et, frame=self.simframe)[:3]
                distance = np.linalg.norm(pos - current_sc_pos)
                if distance < 1e9:
                    x_body = body.radius * x_unit + pos[0]
                    y_body = body.radius * y_unit + pos[1]
                    z_body = body.radius * z_unit + pos[2]
                    if idx in body_surfaces and body_surfaces[idx] is not None:
                        body_surfaces[idx].remove()
                    body_surfaces[idx] = ax.plot_surface(
                        x_body, y_body, z_body,
                        color=body.color, alpha=0.5, rstride=4, cstride=4
                    )
                else:
                    if idx in body_surfaces and body_surfaces[idx] is not None:
                        body_surfaces[idx].remove()
                        body_surfaces[idx] = None

                if idx in body_centers and body_centers[idx] is not None:
                    body_centers[idx].remove()
                body_centers[idx] = ax.scatter(
                    pos[0], pos[1], pos[2], marker="x", color=body.color, alpha=0.5, s=20
                )

                if idx in body_labels and body_labels[idx] is not None:
                    body_labels[idx].remove()
                label_x = pos[0]
                label_y = pos[1]
                label_z = pos[2] + body.radius * 1.05
                body_labels[idx] = ax.text(
                    label_x, label_y, label_z, body.name,
                    color="white", fontsize=9, ha="center", va="bottom"
                )

            ax.set_xlim(current_sc_x - bb_half[0], current_sc_x + bb_half[0])
            ax.set_ylim(current_sc_y - bb_half[0], current_sc_y + bb_half[0])
            ax.set_zlim(current_sc_z - bb_half[0], current_sc_z + bb_half[0])

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

        ani = animation.FuncAnimation(
            fig, update, frames=len(sat_x),
            interval=animate_rate, blit=False, repeat=False
        )

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



def compare_fdata_list(fdata_list, labels, flight_ephem, scraft_id, n_points, end_et):

    # --- Load the necessary SPICE kernels ---
    spice.furnsh(flight_ephem)
    spice.furnsh("ephems/de440.bsp")
    spice.furnsh("ephems/naif0012.tls")
    
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
    
    k = 0
    # --- Process each FData simulation ---
    for fdata in fdata_list:
        # Filter the simulation data to only use times <= end_et
        mask = fdata.pos_arr[:, 3] <= end_et
        if not np.any(mask):
            print(f"Simulation '{fdata.title}' has no data up to the specified end ET {end_et}.")
            continue
        pos_arr_subset = fdata.pos_arr[mask]

        # Calculate mission elapsed times by subtracting the mission's start ET
        et_start = pos_arr_subset[0, 3]
        elapsed_time = pos_arr_subset[:, 3] - et_start

        # Create an array to store true state positions with the same shape as the filtered data.
        true_pos_arr = np.zeros_like(pos_arr_subset)

        # Loop over the filtered simulation times to retrieve the true spacecraft state from SPICE.
        for i in range(pos_arr_subset.shape[0]):
            et = pos_arr_subset[i, 3]
            state, light_time = spice.spkezr(str(scraft_id), et, "J2000", "NONE", fdata.simframe)
            # Convert from km (SPICE default) to m and store into true position array.
            true_pos_arr[i, :3] = state[:3] * 1000
            true_pos_arr[i, 3] = et

        # Compute the error norm (L2 norm) between simulated and SPICE true positions.
        error_norm = np.linalg.norm(pos_arr_subset[:, :3] - true_pos_arr[:, :3], axis=1)
        
        # Compute and print summary error statistics.
        max_error = np.max(error_norm)
        mean_error = np.mean(error_norm)
        final_error = error_norm[-1]
        print(f"Simulation '{fdata.title}':")
        print(f"  Max error:   {max_error:.6f} m")
        print(f"  Mean error:  {mean_error:.6f} m")
        print(f"  Final error: {final_error:.6f} m\n")
        
        # Sample the data to plot only n_points for clarity.
        total_points = len(elapsed_time)
        sample_indices = np.linspace(0, total_points - 1, n_points, dtype=int)
        sampled_elapsed = elapsed_time[sample_indices]
        sampled_error = error_norm[sample_indices]
        
        # Plot the error vs. mission elapsed time for this simulation using the corresponding label.
        plt.plot(sampled_elapsed, sampled_error, label=labels[k])
        k += 1

    

    # --- Finalize the plot ---
    #plt.title("RK4 Position Error Over Mission Elapsed Time (All Simulations)")
    plt.xlabel("Simulation Elapsed Time [s]")
    plt.ylabel("Position Error [m]")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Unload SPICE kernels ---
    spice.unload(flight_ephem)
    spice.unload("ephems/de440.bsp")
    spice.unload("ephems/naif0012.tls")



#Preloaded Simulations to Visualize and Plot (so you don't have to wait for a whole Voyager 2 simulation to run lol)

lro_fdata1  = FData.load_from_file("examples/fdata_lro_h1_em.pkl")  #Lunar Reconnaisance Orbiter Earth-Moon
lro_fdata2  = FData.load_from_file("examples/fdata_lro_h1_fs.pkl")  #Lunar Reconnaisance Orbiter Full-Solar
v2_fdata1   = FData.load_from_file("examples/fdata_v2_h100.pkl")    #Voyager-2 Earth->Jupiter->Saturn

#Uncomment To Run Visualiztion
#lro_fdata1.visualize()
#lro_fdata2.visualize()
#v2_fdata1.visualize(time_warp=100000)

#Sample simulation for the a Molniya Satellite
t_end = 90000 #simulation duration (seconds)
Molniya = SCraft(1600, 0, 0, 320, np.array([0, -2.08e7, 4.16e7, 0, -1.8e3, 0]))
Sim1    = RK4Sim(0.25, Molniya, 0, t_end, "Sample Molniya Simulation (ECI)", "EM") #In Earth Moon System


#Sim1.run()
#Molniya_fdata1 = FData(Sim1)
#Molniya_fdata1.visualize(time_warp=100)

#Sample Simulation for LRO after TLI
LRO_tli_et  = 298636369.18444437
sim_hours   = 144
timestep    = 1
frame       = "EM" #try changing to "FS" for a Sun Centered Inertial Version with all planets' gravities
lro_sample  = SCraft(1016, 900, 0, 320, np.array([10167365.975892201, 1856377.8084626137, -3676233.578828538, 4562.784281644817, 7032.767739953707, 461.32442763548767]))
Sim2        = RK4Sim(timestep, lro_sample, LRO_tli_et, LRO_tli_et+3600*sim_hours, "Sample LRO TLI Simulation", frame) #In Earth Moon System

# Uncomment to run and visualize LRO:
Sim2.run() 
lro_fdata3 = FData(Sim2)
lro_fdata3.visualize(time_warp=1000)





#Old code used to run convergence tests and make graphs etc.


# Sample spacecraft state vector: [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z]
#sat1 = SCraft(100, 80, 1000, 320, np.array([10167365.975892201, 1856377.8084626137, -3676233.578828538, 4562.784281644817, 7032.767739953707, 461.32442763548767]))
#sat1 = SCraft(200, 80, 1000, 320, np.array([9012997.271085758, 1269718.7816287968, 4747449.646182241, 4590.474197406957, 8909.517411247949, 8932.203969847747]))
# h, scraft, t_start, t_end, title, system
# sat1 = SCraft(200, 80, 1000, 320, np.array([-634618575371.3771, 622437168447.8938, 281745034317.3275, -54235.91075183407, -7595.920588155404, -2912.718916307579]))


#sim1 = RK4Sim(1, sat1, 298636369.18444437, 2.9925e8, "Lunar Reconnaissance Orbiter Lunar Injection and Transfer", "EM")
#sim2 = RK4Sim(2, sat1, 298635469, 2.99025e8, "Lunar Reconnaissance Orbiter Lunar Injection and Transfer", "EM")
#sim3 = RK4Sim(4, sat1, 298635469, 2.99025e8, "Lunar Reconnaissance Orbiter Lunar Injection and Transfer", "EM")
#sim4 = RK4Sim(8, sat1, 298635469, 2.99025e8, "Lunar Reconnaissance Orbiter Lunar Injection and Transfer", "EM")
#sim5 = RK4Sim(16, sat1, 298635469, 2.99025e8, "Lunar Reconnaissance Orbiter Lunar Injection and Transfer", "EM")

#sim1 = RK4Sim(100, sat1, -705788498.8171841, -705788798.8171841+3*31557600, "Voyager 2", "FS")

#sim1.run()
#sim2.run()
#sim3.run()
#sim4.run()
#sim5.run()

#fdata1 = FData(sim1)
#fdata2 = FData(sim2)
#fdata3 = FData(sim3)
#fdata4 = FData(sim4)
#fdata5 = FData(sim5)

#fdata1.save_to_file("fdata_lro_h1_2.pkl")
#fdata2.save_to_file("lro_hdata/fdata_lro_h2.pkl")
#fdata3.save_to_file("lro_hdata/fdata_lro_h4.pkl")
#fdata4.save_to_file("lro_hdata/fdata_lro_h8.pkl")
#fdata5.save_to_file("lro_hdata/fdata_lro_h16.pkl")

#fdata1 = FData.load_from_file("fdata_v2_h100.pkl")
#fdata1 = FData.load_from_file("fdata_lro_h1_bv.pkl")
#fdata2 = FData.load_from_file("lro_hdata/fdata_lro_h1.pkl")
#fdata3 = FData.load_from_file("lro_hdata/fdata_lro_h2.pkl")
#fdata4 = FData.load_from_file("lro_hdata/fdata_lro_h4.pkl")
#fdata5 = FData.load_from_file("lro_hdata/fdata_lro_h8.pkl")
#fdata6 = FData.load_from_file("lro_hdata/fdata_lro_h16.pkl")

#fdata1 = FData.load_from_file("fdata_lro_h1_2.pkl")
#fdata2 = FData.load_from_file("fdata_v2_h100.pkl")
#fdata3 = FData.load_from_file("fdata_v2_h1000.pkl")
#fdata1.visualize(time_warp = 100000)

#fdata1.visualize(time_warp = 1000)
###fdata1.compare_velocity("flight_ephems/Voyager_2.m05016u.merged.bsp", -32, 10000000)
###fdata1.compare_acceleration("flight_ephems/Voyager_2.m05016u.merged.bsp", -32, 10000000)
###fdata1.compare_jerk("flight_ephems/Voyager_2.m05016u.merged.bsp", -32, 10000000)

#compare_fdata_list([fdata1, fdata2, fdata3, fdata4, fdata5], ["h = 0.5 s", "h = 1.0 s", "h = 2.0 s", "h = 4.0 s", "h = 8.0 s", "h = 16.0 s"], "flight_ephems/lrorg_2009169_2010001_v01.bsp", -85, 10000, 2.99024e8)
#compare_fdata_list([fdata1, fdata2], ["h=0.5 s", "h=1 s"], "flight_ephems/lrorg_2009169_2010001_v01.bsp", -85, 10000, 2.99024e8)

#compare_fdata_list([fdata1, fdata2], ["Full-Solar", "Earth-Moon"], "flight_ephems/lrorg_2009169_2010001_v01.bsp", -85, 10000, 2.99025e8)
#compare_fdata_list([fdata1, fdata2, fdata3], ["h40", "h100", "h1000"], "flight_ephems/Voyager_2.m05016u.merged.bsp", -32, 100000, -705788798.8171841+4.49*31557600)

#"flight_ephems/Voyager_2.m05016u.merged.bsp"