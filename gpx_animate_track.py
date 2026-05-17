import xml.etree.ElementTree as ET
import cv2
import numpy as np
import ffmpeg
import os
import requests
from PIL import Image

# Third-party helper to cleanly parse and stitch open-source map tiles natively
import staticmaps

custom_markers = [
    (45.432, 6.380, "Col de la Madeleine")
]


class GPXOverlayRenderer:
    """
    GPX → Animated track video with an optional auto-downloaded OpenStreetMap background.
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
            margin=100,
            use_online_map=True,  # Toggle to automatically pull maps online
            easing=True,
            custom_markers=None
    ):
        self.gpx_path = gpx_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.line_color = line_color
        self.line_width = line_width
        self.margin = margin
        self.use_online_map = use_online_map
        self.easing = easing

        self.coords = self.load_gpx()

        # Build map distance arrays
        self.distances = self.compute_distances()
        self.total_distance = self.distances[-1]

        # Initialize background canvas context to query pixel projection metrics
        self.context = staticmaps.Context()
        self.context.set_tile_provider(staticmaps.tile_provider_OSM)

        # Feed the coordinates to context so it can compute bounds automatically
        gpx_line = staticmaps.Line(
            [staticmaps.create_latlng(lat, lon) for lat, lon in self.coords],
            color=staticmaps.TRANSPARENT,
            width=0
        )
        self.context.add_object(gpx_line)

        # Compute layout metrics using the proper public methods
        center, zoom = self.context.determine_center_zoom(self.width - 2 * self.margin, self.height - 2 * self.margin)

        # Fix: Fetch tile_size cleanly from the explicit tile provider object
        self.transformer = staticmaps.Transformer(
            self.width,
            self.height,
            zoom,
            center,
            staticmaps.tile_provider_OSM.tile_size()
        )

        # Map pixel tracks directly using the exact underlying Web Mercator projections
        self.points = self.project_points()

        self.custom_markers = custom_markers or []
        self.marker_points = self.project_markers()
        self.frame_distances = self.precompute_frame_distances()

        # Cache structural background matrix
        self.cached_bg = self.prepare_background()

    def compute_distances(self):
        d = [0]
        total = 0
        for i in range(1, len(self.coords)):
            total += self.haversine(self.coords[i - 1], self.coords[i])
            d.append(total)
        return d

    def project_markers(self):
        markers = []
        # START
        markers.append(("start", self.points[0], 0.0))

        # CUSTOM MARKERS
        for lat, lon, *_ in self.custom_markers:
            closest_idx = 0
            min_d = float('inf')
            for idx, coord in enumerate(self.coords):
                d = self.haversine((lat, lon), coord)
                if d < min_d:
                    min_d = d
                    closest_idx = idx

            trigger_dist = self.distances[closest_idx]
            markers.append(("custom", self.points[closest_idx], trigger_dist))

        # END
        markers.append(("end", self.points[-1], self.total_distance))
        return markers

    def precompute_frame_distances(self):
        total_frames = int(self.duration * self.fps)
        unique_triggers = sorted(list(set([m[2] for m in self.marker_points])))
        pause_len = int(0.8 * self.fps)

        if unique_triggers and (len(unique_triggers) * pause_len >= total_frames * 0.6):
            pause_len = int((total_frames * 0.5) / len(unique_triggers))

        total_pause_frames = len(unique_triggers) * pause_len
        moving_frames = max(1, total_frames - total_pause_frames)

        def get_moving_dist(mf):
            if moving_frames <= 1:
                return self.total_distance
            t_rel = mf / (moving_frames - 1)
            p = self.ease(t_rel)
            return p * self.total_distance

        frame_distances = []
        mf_idx = 0
        next_trigger_idx = 0

        while len(frame_distances) < total_frames:
            if next_trigger_idx < len(unique_triggers):
                target_trigger = unique_triggers[next_trigger_idx]
                current_dist_proposal = get_moving_dist(mf_idx)

                if (target_trigger == 0.0 and mf_idx == 0) or (current_dist_proposal >= target_trigger):
                    actual_pause_frames = min(pause_len, total_frames - len(frame_distances))
                    for _ in range(actual_pause_frames):
                        frame_distances.append(target_trigger)
                    next_trigger_idx += 1
                    continue

            if len(frame_distances) < total_frames:
                frame_distances.append(get_moving_dist(mf_idx))
                mf_idx = min(mf_idx + 1, moving_frames - 1)

        return frame_distances

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

    def project_points(self):
        """
        Uses staticmaps internal mercator transformer to map coordinates to pixels perfectly.
        """
        points = []
        for lat, lon in self.coords:
            latlng = staticmaps.create_latlng(lat, lon)
            x, y = self.transformer.ll2pixel(latlng)
            points.append((int(x), int(y)))
        return points

    def ease(self, t):
        if not self.easing:
            return t
        return 1 - (1 - t) ** 3

    def prepare_background(self):
        if not self.use_online_map:
            return None

        print("Fetching aligned map context background layer from OpenStreetMap...")
        pil_image = self.context.render_pillow(self.width, self.height)

        # Convert Image back to OpenCV array standard (RGBA)
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2RGBA)
        return cv_img

    def background(self):
        if self.cached_bg is not None:
            return self.cached_bg.copy()
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def render_frame(self, t):
        frame_idx = min(int(round(t * self.fps)), len(self.frame_distances) - 1)
        target_dist = self.frame_distances[frame_idx]

        visible = []
        for i, d in enumerate(self.distances):
            if d <= target_dist:
                visible.append(self.points[i])
            else:
                break

        if len(visible) < 2:
            visible = self.points[:2]

        frame = self.background()

        if len(visible) > 1:
            cv2.polylines(frame, [np.array(visible)], False, self.line_color, self.line_width, lineType=cv2.LINE_AA)

        if visible:
            x, y = visible[-1]
            cv2.circle(frame, (x, y), self.line_width + 2, self.line_color, -1, lineType=cv2.LINE_AA)

        for mtype, (x, y), trigger_dist in self.marker_points:
            if target_dist >= trigger_dist:
                radius = 12 if mtype in ("start", "end") else 10
                cv2.circle(frame, (x, y), radius, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (x, y), radius, (0, 0, 0, 255), 2, lineType=cv2.LINE_AA)

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
        for i in range(total_frames):
            t = i / self.fps
            frame = self.render_frame(t)
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()
        print(f"Render Complete → {self.output_path}")


if __name__ == "__main__":
    renderer = GPXOverlayRenderer(
        gpx_path="cols.gpx",
        output_path="gpx_overlay.mov",
        width=1080,
        height=1920,
        fps=30,
        duration=10,
        line_color=(0, 102, 100, 255),  # Clear blue line
        line_width=6,
        margin=150,
        use_online_map=True,
        easing=True,
        custom_markers=custom_markers,
    )
    renderer.render()