import xml.etree.ElementTree as ET
import cv2
import numpy as np
import ffmpeg
import os
import sys
import requests
from PIL import Image

# Third-party helper to cleanly parse and stitch open-source map tiles natively
import staticmaps

custom_markers = [
    (45.432, 6.380, "Col de la Madeleine")
]


class GPXOverlayRenderer:
    """
    GPX → Animated track video with an elastic camera tracking (parallax flyover) effect.
    """

    def __init__(
            self,
            gpx_path,
            output_path="gpx_overlay.mov",
            width=1080,
            height=1920,
            fps=30,
            duration=10,
            line_color=(255, 255, 255, 255),  # BGRA
            line_width=6,
            use_online_map=True,
            easing=True,
            custom_markers=None,
            flyover=True,
            flyover_zoom=14,
            camera_lag=0.12  # NEW: 1.0 = locked rigid. Lower values (0.05 - 0.2) create a lazy, fluid camera lag
    ):
        self.gpx_path = gpx_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.line_color = line_color
        self.line_width = line_width
        self.use_online_map = use_online_map
        self.easing = easing
        self.flyover = flyover
        self.flyover_zoom = flyover_zoom
        self.camera_lag = camera_lag

        self.coords = self.load_gpx()

        # Build map distance arrays
        self.distances = self.compute_distances()
        self.total_distance = self.distances[-1]

        # Use a safe, active tile server asset configuration
        self.tile_provider = staticmaps.TileProvider(
            name="carto_dark",
            url_pattern="https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
            max_zoom=20
        )

        self.context = staticmaps.Context()
        self.context.set_tile_provider(self.tile_provider)

        if not self.flyover:
            gpx_line = staticmaps.Line(
                [staticmaps.create_latlng(lat, lon) for lat, lon in self.coords],
                color=staticmaps.TRANSPARENT,
                width=0
            )
            self.context.add_object(gpx_line)
            center, zoom = self.context.determine_center_zoom(self.width, self.height)
            self.transformer = staticmaps.Transformer(
                self.width, self.height, zoom, center, self.tile_provider.tile_size()
            )
            self.points = [self.transformer.ll2pixel(staticmaps.create_latlng(lat, lon)) for lat, lon in self.coords]

        self.custom_markers = custom_markers or []

        # Precompute target metrics along with the camera paths
        self.frame_distances = self.precompute_frame_distances()
        self.camera_centers = self.precompute_camera_paths()

    def compute_distances(self):
        d = [0]
        total = 0
        for i in range(1, len(self.coords)):
            total += self.haversine(self.coords[i - 1], self.coords[i])
            d.append(total)
        return d

    def get_coord_at_distance(self, target_dist):
        """Finds the interpolation lat/lon position at an exact distance point."""
        if target_dist <= 0:
            return self.coords[0]
        if target_dist >= self.total_distance:
            return self.coords[-1]

        for i, d in enumerate(self.distances):
            if d >= target_dist:
                d_prev = self.distances[i - 1]
                d_next = d
                segment_pct = (target_dist - d_prev) / (d_next - d_prev or 1e-9)

                lat_prev, lon_prev = self.coords[i - 1]
                lat_next, lon_next = self.coords[i]

                lat = lat_prev + segment_pct * (lat_next - lat_prev)
                lon = lon_prev + segment_pct * (lon_next - lon_prev)
                return (lat, lon)
        return self.coords[-1]

    def precompute_frame_distances(self):
        total_frames = int(self.duration * self.fps)
        moving_frames = total_frames

        def get_moving_dist(mf):
            if moving_frames <= 1:
                return self.total_distance
            t_rel = mf / (moving_frames - 1)
            p = self.ease(t_rel)
            return p * self.total_distance

        return [get_moving_dist(i) for i in range(total_frames)]

    def precompute_camera_paths(self):
        """NEW: Pre-calculates an elastic trailing camera path path based on cursor inertia."""
        total_frames = int(self.duration * self.fps)
        centers = []

        # Start camera exactly at the beginning point
        cam_lat, cam_lon = self.coords[0]

        for i in range(total_frames):
            target_dist = self.frame_distances[i]
            cursor_lat, cursor_lon = self.get_coord_at_distance(target_dist)

            if i > 0:
                # Linearly interpolate towards cursor using our lag dampener coefficient
                cam_lat = cam_lat * (1 - self.camera_lag) + cursor_lat * self.camera_lag
                cam_lon = cam_lon * (1 - self.camera_lag) + cursor_lon * self.camera_lag

            centers.append((cam_lat, cam_lon))
        return centers

    def load_gpx(self):
        tree = ET.parse(self.gpx_path)
        root = tree.getroot()
        coords = []
        for pt in root.findall(".//{*}trkpt"):
            coords.append((float(pt.attrib["lat"]), float(pt.attrib["lon"])))
        if len(coords) < 2:
            raise ValueError("GPX must contain at least 2 points")
        return coords

    def haversine(self, p1, p2):
        from math import radians, sin, cos, sqrt, atan2
        lat1, lon1 = p1
        lat2, lon2 = p2
        R = 6371000
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def ease(self, t):
        if not self.easing:
            return t
        return 1 - (1 - t) ** 3

    def render_frame(self, t):
        frame_idx = min(int(round(t * self.fps)), len(self.frame_distances) - 1)
        target_dist = self.frame_distances[frame_idx]

        # Extract current moving position coords
        cursor_lat, cursor_lon = self.get_coord_at_distance(target_dist)

        if self.flyover:
            # FIX: Fetch the lazy camera center coordinate instead of the literal cursor position
            cam_lat, cam_lon = self.camera_centers[frame_idx]
            current_center = staticmaps.create_latlng(cam_lat, cam_lon)

            transformer = staticmaps.Transformer(
                self.width, self.height, self.flyover_zoom, current_center, self.tile_provider.tile_size()
            )
            # Map full trace coordinates layout points on this frame's perspective space
            frame_points = []
            for lat, lon in self.coords:
                x, y = transformer.ll2pixel(staticmaps.create_latlng(lat, lon))
                frame_points.append((int(x), int(y)))
        else:
            transformer = self.transformer
            frame_points = self.points

        # Slice the visible portion of the track up to the current distance
        visible_points = []
        for i, d in enumerate(self.distances):
            if d <= target_dist:
                visible_points.append(frame_points[i])
            else:
                # Interpolate the exact cutting pixel edge point
                lat_edge, lon_edge = self.get_coord_at_distance(target_dist)
                x_edge, y_edge = transformer.ll2pixel(staticmaps.create_latlng(lat_edge, lon_edge))
                visible_points.append((int(x_edge), int(y_edge)))
                break

        if len(visible_points) < 2:
            visible_points = frame_points[:2]

        # Fetch and stitch the custom tile window viewport configuration
        if self.use_online_map:
            if self.flyover:
                # Update persistent context to match our lagging camera coordinate anchor
                cam_lat, cam_lon = self.camera_centers[frame_idx]
                self.context.set_center(staticmaps.create_latlng(cam_lat, cam_lon))
                self.context.set_zoom(self.flyover_zoom)
            pil_image = self.context.render_pillow(self.width, self.height)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2RGBA)
        else:
            frame = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Draw the trailing partial path track vector overlay
        if len(visible_points) > 1:
            cv2.polylines(frame, [np.array(visible_points)], False, self.line_color, self.line_width,
                          lineType=cv2.LINE_AA)

        # Draw the target trace cursor dot directly over its active localized coordinate space
        # FIX: The pixel position will now dynamically wander across the viewport!
        cx, cy = transformer.ll2pixel(staticmaps.create_latlng(cursor_lat, cursor_lon))
        cv2.circle(frame, (int(cx), int(cy)), self.line_width + 4, self.line_color, -1, lineType=cv2.LINE_AA)

        # Render custom markers if they fall inside our active zoom window dimensions
        for lat, lon, *_ in self.custom_markers:
            mx, my = transformer.ll2pixel(staticmaps.create_latlng(lat, lon))
            if 0 <= mx <= self.width and 0 <= my <= self.height:
                cv2.circle(frame, (int(mx), int(my)), 10, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (int(mx), int(my)), 10, (0, 0, 0, 255), 2, lineType=cv2.LINE_AA)

        return frame

    def render(self):
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgra', s=f'{self.width}x{self.height}', r=self.fps)
            .output(self.output_path, vcodec='prores_ks', profile=4, pix_fmt='yuva444p10le', r=self.fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        total_frames = int(self.duration * self.fps)
        print("\nStarting video render loop...")

        for i in range(total_frames):
            t = i / self.fps
            frame = self.render_frame(t)
            process.stdin.write(frame.tobytes())

            current_frame = i + 1
            percent = (current_frame / total_frames) * 100
            bar_length = 30
            filled_length = int(round(bar_length * current_frame / float(total_frames)))
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            sys.stdout.write(f"\rProcessing: [{bar}] {current_frame}/{total_frames} ({percent:.1f}%)")
            sys.stdout.flush()

        process.stdin.close()
        process.wait()
        print(f"\n\nRender Complete → {self.output_path}\n")


if __name__ == "__main__":
    renderer = GPXOverlayRenderer(
        gpx_path="cols.gpx",
        output_path="gpx_overlay.mov",
        width=1080,
        height=1920,
        fps=30,
        duration=10,
        line_color=(0, 102, 255, 255),
        line_width=6,
        use_online_map=True,
        easing=True,
        custom_markers=custom_markers,
        flyover=True,
        flyover_zoom=14,
        camera_lag=0.12
        # Play with this! Lower value (e.g. 0.08) = more cursor movement. Higher value (e.g. 0.25) = tighter tracking.
    )
    renderer.render()