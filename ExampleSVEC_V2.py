import spiceypy as spice

def main():
    # Define file paths (adjust these paths as needed).
    de_kernel   = "ephems/de440.bsp"                     # Planetary ephemeris kernel
    voy2_kernel = "flight_ephems/Voyager_2.m05016u.merged.bsp"                        # Voyager 2 SPICE kernel
    leap_kernel = "naif0012.tls"                        # Leap-seconds kernel

    # Load the kernels.
    try:
        spice.furnsh(de_kernel)
        spice.furnsh(voy2_kernel)
        spice.furnsh(leap_kernel)
    except Exception as e:
        print("Error loading kernels:", e)
        return

    # Set the epoch just after the Star-37E burn (approximately 15:00 UTC on 1977-08-20).
    epoch_str = "1977-08-20T15:32:33"
    et = spice.str2et(epoch_str)

    # Print names and IDs in the Voyager 2 kernel
    ids = spice.spkobj(voy2_kernel)
    for obj_id in ids:
        try:
            obj_name = spice.bodc2n(obj_id)
        except:
            obj_name = "Unknown"
        print(f"Object ID: {obj_id}, Name: {obj_name}")

    # Set the target, reference frame, and observer.
    target    = "-32"     # Voyager 2's SPICE ID
    ref_frame = "J2000"
    observer  = "EARTH"

    try:
        # Retrieve the state vector.
        state, lt = spice.spkezr(target, et, ref_frame, "NONE", observer)
    except Exception as e:
        print("Error retrieving state vector:", e)
        spice.unload(de_kernel)
        spice.unload(voy2_kernel)
        spice.unload(leap_kernel)
        return

    # Convert state vector from km and km/s to meters and m/s.
    state_si = [elem * 1000 for elem in state]

    print(f"Epoch (ET): {et} seconds past J2000 ({epoch_str})")
    print("Voyager 2 state vector (J2000, relative to Earth):")
    print("Position (m):", state_si[:3])
    print("Velocity (m/s):", state_si[3:])

    # Unload the kernels.
    spice.unload(de_kernel)
    spice.unload(voy2_kernel)
    spice.unload(leap_kernel)

if __name__ == "__main__":
    main()
