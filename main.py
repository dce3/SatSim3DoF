import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice

#Newton's Gravitational Constant
NGC = 6.67430e-11

#Load the ephemeris kernel
EPHEM_KERNEL_ADDRESS = "ephems/de440.bsp"
spice.furnsh(EPHEM_KERNEL_ADDRESS)
spice.furnsh("naif0012.tls") #This one handles the leapseconds

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
        for i in range(0, self.hcount):
            self.pos_arr[i,:] = self.tstep(0, et)
            et += self.h





# Define Earth and spacecraft, then run the simulation
earth = CBody("Earth", "EARTH", 5.9722e24, 6378137)
# Note: The spacecraft state vector should contain position (first three values)
# and velocity (last three values). Here we assume the units are consistent (meters, m/s).
sat1 = SCraft(100, 80, 1000, 320, np.array([0, earth.radius + 400000, 0, 10000, 0, 1200]))

# Set simulation parameters: timestep of 0.1 seconds, running from t=0 to t=20 seconds.
sim1 = RK4Sim(0.1, [earth], [sat1], 0, 45000)
sim1.run()
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
