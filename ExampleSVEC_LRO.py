"""
Script to retrieve the Lunar Reconnaissance Orbiter (LRO) state vector using the BSP file:
lrorg_2009169_2010001_v01.bsp.
According to published launch reports, the LRO’s Trans-Lunar Injection (TLI) burn ended at about 22:16 UT
on June 18, 2009. For this example, we chose an ET immediately following the TLI (22:16:43 UT).
This script also loads a planetary ephemeris kernel (de440.bsp) to compute Earth's state.
Ensure that all required kernels are available in your working directory.
"""

import spiceypy as spice

def main():
    # Define file paths (adjust these paths as needed).
    de_kernel   = "ephems/de440.bsp"                     # Planetary ephemeris kernel (provides Earth's state)
    lro_kernel  = "flight_ephems/lrorg_2009169_2010001_v01.bsp"          # LRO BSP kernel (covers 2009-06-18 to 2010-01-01)
    leap_kernel = "naif0012.tls"                # Leap-seconds kernel

    # Load the kernels.
    try:
        spice.furnsh(de_kernel)
        spice.furnsh(lro_kernel)
        spice.furnsh(leap_kernel)
    except Exception as e:
        print("Error loading kernels:", e)
        return

    # Choose an epoch that falls within the BSP file's coverage.
    # Here we choose an epoch just after the TLI ended (approximately 22:16:43 UT on 2009-06-18).
    epoch_str = "2009-06-18T22:16:43"
    et = spice.str2et(epoch_str)

    # Set the target, reference frame, and observer.
    # For LRO, the SPICE ID is typically set to -120.
    target    = "-85"
    ref_frame = "J2000"
    observer  = "EARTH"  # Change to 'MOON' if you prefer a selenocentric state vector

    try:
        # Retrieve the state vector.
        # 'state' is a six-element array: [x, y, z, vx, vy, vz] in km and km/s.
        state, lt = spice.spkezr(target, et, ref_frame, "NONE", observer)
    except Exception as e:
        print("Error retrieving state vector:", e)
        spice.unload(de_kernel)
        spice.unload(lro_kernel)
        spice.unload(leap_kernel)
        return

    # Convert state vector from km and km/s to meters and m/s.
    state_si = [elem * 1000 for elem in state]

    print(f"Epoch (ET): {et} seconds past J2000 ({epoch_str})")
    print("LRO state vector (J2000, relative to Earth):")
    print("Position (m):", state_si[:3])
    print("Velocity (m/s):", state_si[3:])

    # Unload the kernels.
    spice.unload(de_kernel)
    spice.unload(lro_kernel)
    spice.unload(leap_kernel)

if __name__ == "__main__":
    main()
