import spiceypy as spice
import numpy as np

zarr = np.zeros((10, 3))

print(zarr, "\n")


# Load the necessary SPICE kernels
spice.furnsh("naif0012.tls")
spice.furnsh("ephems/de430_1850-2150.bsp")

# Specify the observation time (example: January 1, 2025 at 12:00:00 UTC)
time_str = "1992-01-01T12:00:00"
et = spice.str2et(time_str)

print(et)

spk_ids = spice.spkobj("ephems/de440.bsp")
print("Object IDs and names in the kernel:")
for i in range(spk_ids.card):
    obj_id = spk_ids[i]
    obj_name = spice.bodc2s(obj_id)
    print(f"ID {obj_id}: {obj_name}")


# Compute the state (position and velocity) of Mars relative to the Sun in the J2000 frame.
# The returned state vector has 6 components: first three are position (km) and the last three are velocity (km/s)
state, light_time = spice.spkezr("MARS BARYCENTER", et, "J2000", "NONE", "SUN")

# Output the results
print(f"State of Mars relative to the Sun at {time_str}:")
print("Position (km):", state[:3])
print("Velocity (km/s):", state[3:])
print(state)

# Unload the kernels after computations to free resources
spice.unload("naif0012.tls")
spice.unload("de440.bs")
