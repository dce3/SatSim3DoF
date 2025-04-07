import math
import json
import webbrowser
import os

# -------------------------------
# Example: Generate orbit data
# -------------------------------
# For demonstration, we create a circular orbit in the equatorial plane.
# Replace this with your own simulation data (e.g., SPICE-derived positions).
num_points = 360
orbit_positions = []  # Each entry: [longitude (deg), latitude (deg), altitude (meters)]
orbit_altitude = 400000.0  # e.g. 400 km altitude

for i in range(num_points):
    # For a circular orbit, longitude changes uniformly.
    lon = i  # degrees (0 to 359)
    lat = 0  # equatorial orbit
    orbit_positions.append([lon, lat, orbit_altitude])

# -------------------------------
# Create HTML content with embedded CesiumJS
# -------------------------------
# (Replace 'your_cesium_ion_access_token_here' with your actual Cesium Ion access token.)
html_content = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Cesium Animated Earth with Orbit</title>
    <script src="https://cesium.com/downloads/cesiumjs/releases/1.102/Build/Cesium/Cesium.js"></script>
    <link rel="stylesheet" href="https://cesium.com/downloads/cesiumjs/releases/1.102/Build/Cesium/Widgets/widgets.css">
    <style>
      html, body, #cesiumContainer {{
         width: 100%; 
         height: 100%; 
         margin: 0; 
         padding: 0; 
         overflow: hidden;
      }}
    </style>
  </head>
  <body>
    <div id="cesiumContainer"></div>
    <script>
      // Set your Cesium Ion access token (get one from https://cesium.com/ion/)
      Cesium.Ion.defaultAccessToken = 'your_cesium_ion_access_token_here';

      // Create the Cesium Viewer with timeline and animation controls enabled.
      const viewer = new Cesium.Viewer('cesiumContainer', {{
        timeline: true,
        animation: true,
        baseLayerPicker: true
      }});

      // Embed the orbit data computed in Python.
      const orbitPositions = {json.dumps(orbit_positions)};

      // Flatten the orbit data for Cesium's fromDegreesArrayHeights.
      const flatPositions = [];
      for (let i = 0; i < orbitPositions.length; i++) {{
        flatPositions.push(orbitPositions[i][0]);
        flatPositions.push(orbitPositions[i][1]);
        flatPositions.push(orbitPositions[i][2]);
      }}

      // Create the orbit path entity.
      const orbitEntity = viewer.entities.add({{
        name: 'Orbit Path',
        polyline: {{
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(flatPositions),
          width: 3,
          material: Cesium.Color.RED
        }}
      }});

      // Add a spacecraft entity that will follow the orbit.
      const spacecraft = viewer.entities.add({{
        name: 'Spacecraft',
        position: Cesium.Cartesian3.fromDegrees(orbitPositions[0][0], orbitPositions[0][1], orbitPositions[0][2]),
        point: {{
          pixelSize: 10,
          color: Cesium.Color.YELLOW
        }}
      }});

      // Enable animation using Cesium's clock.
      viewer.clock.shouldAnimate = true;
      // Adjust the clock multiplier to control animation speed.
      viewer.clock.multiplier = 50;

      let currentIndex = 0;
      // Animate the spacecraft along the orbit.
      viewer.clock.onTick.addEventListener(function(clock) {{
        currentIndex = (currentIndex + 1) % orbitPositions.length;
        const pos = orbitPositions[currentIndex];
        spacecraft.position = Cesium.Cartesian3.fromDegrees(pos[0], pos[1], pos[2]);
      }});

      // Cesium's viewer automatically renders the rotating globe.
      // You can zoom to your entities:
      viewer.zoomTo(viewer.entities);
    </script>
  </body>
</html>
"""

# -------------------------------
# Write the HTML content to a file and open it in the browser.
# -------------------------------
output_filename = "cesium_animation.html"
with open(output_filename, "w") as f:
    f.write(html_content)

webbrowser.open("file://" + os.path.abspath(output_filename))
