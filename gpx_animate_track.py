import xml.etree.ElementTree as ET
import cv2
import numpy as np
import ffmpeg  # Directly controls FFmpeg for perfect ProRes 4444 encoding

custom_markers = [
    (45.432, 6.380, "Col de la Madeleine")
]


class GPXOverlayRenderer:
    """
    GPX → Animated route video with a rock-solid transparent ProRes alpha channel (.mov)
    """

    def __init__(
            self,
            gpx_path,
            output_path="gpx_overlay.mov",  # ProRes requires a .mov container
            width=1080,
            height=1920,
            fps=30,
            duration=10,
            line_color=(255, 255, 255, 255),  # BGRA format
            line_width=6,
            margin=100,
            background_image=None,
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
        self.background_image = background_image
        self.easing = easing

        self.coords = self.load_gpx()
        self.points = self.project_points()

        self.distances = self.compute_distances()
        self.total_distance = self.distances[-1]

        self.custom_markers = custom_markers or []
        self.marker_points = self.project_markers()

        self.frame_distances = self.precompute_frame_distances()

    def compute_distances(self):
        d = [0]
        total = 0
        for i in range(1, len(self.coords)):
            total += self.haversine(self.coords[i - 1], self.coords[i])
            d.append(total)
        return d

    def project_markers(self):
        lats = [c[0] for c in self.coords]
        lons = [c[1] for c in self.coords]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        lat_range = max_lat - min_lat or 1e-9
        lon_range = max_lon - min_lon or 1e-9

        usable_w = self.width - 2 * self.margin
        usable_h = self.height - 2 * self.margin

        scale_x = usable_w / lon_range
        scale_y = usable_h / lat_range
        scale = min(scale_x, scale_y)

        map_w = lon_range * scale
        map_h = lat_range * scale

        offset_x = (self.width - map_w) / 2
        offset_y = (self.height - map_h) / 2

        markers = []

        # START
        lat, lon = self.coords[0]
        x = (lon - min_lon) * scale + offset_x
        y = (max_lat - lat) * scale + offset_y
        markers.append(("start", (int(x), int(y)), 0.0))

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
            x = (lon - min_lon) * scale + offset_x
            y = (max_lat - lat) * scale + offset_y
            markers.append(("custom", (int(x), int(y)), trigger_dist))

        # END
        lat, lon = self.coords[-1]
        x = (lon - min_lon) * scale + offset_x
        y = (max_lat - lat) * scale + offset_y
        markers.append(("end", (int(x), int(y)), self.total_distance))

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
            coords.append((
                float(pt.attrib["lat"]),
                float(pt.attrib["lon"])
            ))

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
        lats = [c[0] for c in self.coords]
        lons = [c[1] for c in self.coords]

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        lat_range = max_lat - min_lat or 1e-9
        lon_range = max_lon - min_lon or 1e-9

        usable_w = self.width - 2 * self.margin
        usable_h = self.height - 2 * self.margin

        scale_x = usable_w / lon_range
        scale_y = usable_h / lat_range
        scale = min(scale_x, scale_y)

        map_w = lon_range * scale
        map_h = lat_range * scale

        offset_x = (self.width - map_w) / 2
        offset_y = (self.height - map_h) / 2

        points = []
        for lat, lon in self.coords:
            x = (lon - min_lon) * scale + offset_x
            y = (max_lat - lat) * scale + offset_y
            points.append((int(x), int(y)))

        return points

    def ease(self, t):
        if not self.easing:
            return t
        return 1 - (1 - t) ** 3

    def background(self):
        # 4-channel transparent base (B, G, R, Alpha=0)
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

        # Draw transparent route line
        if len(visible) > 1:
            cv2.polylines(
                frame,
                [np.array(visible)],
                False,
                self.line_color,
                self.line_width,
                lineType=cv2.LINE_AA,
            )

        # Moving active "head" dot
        if visible:
            x, y = visible[-1]
            cv2.circle(
                frame,
                (x, y),
                self.line_width + 2,
                self.line_color,
                -1,
                lineType=cv2.LINE_AA,
            )

        # WHITE MARKERS
        for mtype, (x, y), trigger_dist in self.marker_points:
            if target_dist >= trigger_dist:
                color = (255, 255, 255, 255)
                radius = 15 if mtype in ("start", "end") else 15

                cv2.circle(
                    frame,
                    (x, y),
                    radius,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        return frame

    def render(self):
        # Build an FFmpeg process stream that expects raw BGRA image data via pipe input
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgra', s=f'{self.width}x{self.height}', r=self.fps)
            .output(
                self.output_path,
                vcodec='prores_ks',  # FFmpeg's high-fidelity ProRes profile encoder
                profile=4,  # Profile '4' forces ProRes 4444 (XQ/Standard Alpha)
                pix_fmt='yuva444p10le',  # Enforces a 10-bit YUV format containing a discrete Alpha channel
                r=self.fps
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        total_frames = int(self.duration * self.fps)
        for i in range(total_frames):
            t = i / self.fps
            frame = self.render_frame(t)

            # Write bytes directly into the FFmpeg buffer pipe
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()
        print(f"Saved transparent ProRes video → {self.output_path}")


if __name__ == "__main__":
    renderer = GPXOverlayRenderer(
        gpx_path="cols.gpx",
        output_path="gpx_overlay.mov",  # ProRes files must use a .mov extension
        width=1080,
        height=1920,
        fps=30,
        duration=10,
        line_color=(255, 255, 255, 255),
        line_width=6,
        margin=120,
        background_image=None,
        easing=True,
        custom_markers=custom_markers,
    )

    renderer.render()