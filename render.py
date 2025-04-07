# render.py
import json
import webbrowser
import os

def render_orbit(orbit_positions, cesium_token):
    """
    Generates an HTML file with CesiumJS that displays an animated Earth (with black background)
    and an orbit path based on the provided orbit_positions. Each orbit position should be a list:
    [longitude (deg), latitude (deg), altitude (meters)].
    """
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
         background-color: black;
      }}
      .cesium-viewer .cesium-sky-box {{
          background: black;
      }}
    </style>
  </head>
  <body>
    <div id="cesiumContainer"></div>
    <script>
      // Set your Cesium Ion access token.
      Cesium.Ion.defaultAccessToken = '{cesium_token}';
      
      // Create the Cesium Viewer.
      const viewer = new Cesium.Viewer('cesiumContainer', {{
        timeline: true,
        animation: true,
        baseLayerPicker: true,
        scene3DOnly: true,
        skyBox: new Cesium.SkyBox({{
          sources: {{
            positiveX: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAc/INeUAAAAASUVORK5CYII=',
            negativeX: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAc/INeUAAAAASUVORK5CYII=',
            positiveY: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAc/INeUAAAAASUVORK5CYII=',
            negativeY: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAc/INeUAAAAASUVORK5CYII=',
            positiveZ: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAc/INeUAAAAASUVORK5CYII=',
            negativeZ: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAACnej3aAAAAAXRSTlMAQObYZgAAAApJREFUCNdjYAAAAAIAAc/INeUAAAAASUVORK5CYII='
          }}
        }})
      }});

      // Embed orbit data from Python.
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

      // Animate the spacecraft along the orbit.
      viewer.clock.shouldAnimate = true;
      viewer.clock.multiplier = 50;
      let currentIndex = 0;
      viewer.clock.onTick.addEventListener(function(clock) {{
        currentIndex = (currentIndex + 1) % orbitPositions.length;
        const pos = orbitPositions[currentIndex];
        spacecraft.position = Cesium.Cartesian3.fromDegrees(pos[0], pos[1], pos[2]);
      }});

      // Zoom out to view all entities.
      viewer.zoomTo(viewer.entities);
    </script>
  </body>
</html>
"""
    output_filename = "render.html"
    with open(output_filename, "w") as f:
        f.write(html_content)
    webbrowser.open("file://" + os.path.abspath(output_filename))
