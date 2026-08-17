import xml.etree.ElementTree as ET
import cv2
import numpy as np
import ffmpeg

custom_markers = [
    (45.432, 6.380, "Col de la Madeleine")
]


class GPXElevationRenderer:
    """
    GPX → Animated elevation profile with synced marker dots on the bottom third
    of a transparent ProRes alpha channel (.mov).
    """

    def __init__(
            self,
            gpx_path,
            output_path="gpx_elevation.mov",
            width=1080,
            height=1920,
            fps=30,
            duration=10,
            line_color=(255, 255, 255, 255),  # BGRA
            line_width=6,
            margin=120,
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
        self.easing = easing

        # 1. Load data
        self.coords, self.elevations = self.load_gpx_data()

        # 2. Distance metrics
        self.distances = self.compute_distances()
        self.total_distance = self.distances[-1]

        # 3. Calculate marker distances & setup frame tracking
        self.custom_markers = custom_markers or []
        self.marker_data = self.calculate_marker_milestones()
        self.frame_distances = self.precompute_frame_distances()

        # 4. Map dataset layout to pixel coordinates for the bottom third window
        self.elevation_points = self.project_elevation_profile()

        # 5. Pre-project the static marker pixel locations onto the profile
        self.marker_points = self.project_marker_dots()

    def load_gpx_data(self):
        tree = ET.parse(self.gpx_path)
        root = tree.getroot()

        coords = []
        elevations = []
        for pt in root.findall(".//{*}trkpt"):
            coords.append((
                float(pt.attrib["lat"]),
                float(pt.attrib["lon"])
            ))
            ele_node = pt.find("{*}ele")
            elevations.append(float(ele_node.text) if ele_node is not None else 0.0)

        if len(coords) < 2:
            raise ValueError("GPX must contain at least 2 points")

        return coords, elevations

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

    def compute_distances(self):
        d = [0]
        total = 0
        for i in range(1, len(self.coords)):
            total += self.haversine(self.coords[i - 1], self.coords[i])
            d.append(total)
        return d

    def calculate_marker_milestones(self):
        """
        Finds the exact distance milestone index and coordinates for all markers.
        """
        markers = [("start", 0.0, 0)]

        for lat, lon, name in self.custom_markers:
            closest_idx = 0
            min_d = float('inf')
            for idx, coord in enumerate(self.coords):
                d = self.haversine((lat, lon), coord)
                if d < min_d:
                    min_d = d
                    closest_idx = idx
            markers.append(("custom", self.distances[closest_idx], closest_idx))

        markers.append(("end", self.total_distance, len(self.coords) - 1))
        return markers

    def precompute_frame_distances(self):
        total_frames = int(self.duration * self.fps)
        unique_triggers = sorted(list(set([dist for _, dist, _ in self.marker_data])))

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

    def project_elevation_profile(self):
        """
        Maps the data so it fits perfectly edge-to-edge (0 to 1080) on the X-axis,
        and increases the vertical Y-axis height to make peaks more pronounced.
        """
        min_ele, max_ele = min(self.elevations), max(self.elevations)
        ele_range = max_ele - min_ele or 1e-9

        # NEW BOUNDARIES: Expand chart vertically (from 60% down to 95% of screen)
        chart_top = int(self.height * 0.60)  # Pushed higher up the screen
        chart_bottom = int(self.height * 0.95)  # Dropped closer to the absolute bottom edge
        chart_height = chart_bottom - chart_top

        # Perfect Edge-to-Edge scaling factor (No side margins used for X calculation)
        scale_x = self.width / self.total_distance
        scale_y = chart_height / ele_range

        points = []
        for dist, ele in zip(self.distances, self.elevations):
            x = dist * scale_x  # Starts exactly at 0 and finishes exactly at 1080
            y = chart_bottom - ((ele - min_ele) * scale_y)
            points.append((int(x), int(y)))

        return points

    def project_marker_dots(self):
        """
        Extracts the precise pre-calculated chart coordinates for each marker.
        """
        markers = []
        for mtype, dist, idx in self.marker_data:
            # Pull the pre-scaled X/Y point from our profile mapping
            pixel_pt = self.elevation_points[idx]
            markers.append((mtype, pixel_pt, dist))
        return markers

    def ease(self, t):
        if not self.easing:
            return t
        return 1 - (1 - t) ** 3

    def background(self):
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def render_frame(self, t):
        frame_idx = min(int(round(t * self.fps)), len(self.frame_distances) - 1)
        target_dist = self.frame_distances[frame_idx]

        visible_indices = [i for i, d in enumerate(self.distances) if d <= target_dist]
        if len(visible_indices) < 2:
            visible_indices = [0, 1]

        frame = self.background()
        visible_ele_pts = [self.elevation_points[idx] for idx in visible_indices]

        if len(visible_ele_pts) > 1:
            chart_bottom = int(self.height * 0.95)

            # 1. Semi-transparent filled area under the line
            poly_points = (
                    [(visible_ele_pts[0][0], chart_bottom)] +
                    visible_ele_pts +
                    [(visible_ele_pts[-1][0], chart_bottom)]
            )
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(poly_points)], (255, 255, 255, 35))
            cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, dst=frame)

            # 2. Main solid white profile line
            cv2.polylines(frame, [np.array(visible_ele_pts)], False, self.line_color, self.line_width,
                          lineType=cv2.LINE_AA)

        # 3. Dynamic Marker Dots (Rendered once line passes their milestone distance)
        for mtype, (mx, my), trigger_dist in self.marker_points:
            if target_dist >= trigger_dist:
                radius = 12 if mtype in ("start", "end") else 8
                cv2.circle(
                    frame,
                    (mx, my),
                    radius,
                    (255, 255, 255, 255),  # Opaque White Dot
                    -1,
                    lineType=cv2.LINE_AA
                )

        # 4. Animated progress slider head riding on the profile crest
        ele_head_x, ele_head_y = visible_ele_pts[-1]
        cv2.circle(frame, (ele_head_x, ele_head_y), self.line_width + 3, self.line_color, -1, lineType=cv2.LINE_AA)

        return frame

    def render(self):
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgra', s=f'{self.width}x{self.height}', r=self.fps)
            .output(
                self.output_path,
                vcodec='prores_ks',
                profile=4,
                pix_fmt='yuva444p10le',
                r=self.fps
            )
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
        print(f"Saved transparent profile with marker dots → {self.output_path}")


if __name__ == "__main__":
    renderer = GPXElevationRenderer(
        gpx_path="belledonne.gpx",
        output_path="belledonne.mov",
        width=1080,
        height=1920,
        fps=30,
        duration=10,
        line_color=(255, 255, 255, 255),
        line_width=6,
        margin=120,
        easing=True,
        custom_markers=custom_markers,
    )

    renderer.render()