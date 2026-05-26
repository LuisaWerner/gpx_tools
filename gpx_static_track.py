import xml.etree.ElementTree as ET
import cv2
import numpy as np
import staticmaps
from PIL import Image  # Used for Instagram-compatible PNG encoding

custom_markers = [
    (45.432, 6.380, "Col de la Madeleine")
]


class GPXStaticRenderer:
    """
    GPX → Transparent Static PNG overlay optimized for Instagram Stories/Reels.
    """

    def __init__(
            self,
            gpx_path,
            output_path="gpx_instagram_overlay.png",
            width=1080,
            height=1920,
            line_color=(255, 255, 255, 255),  # BGRA (White)
            line_width=6,
            margin=100,
            use_online_map=False,  # MUST BE FALSE FOR TRANSPARENT OVERLAY
            custom_markers=None
    ):
        self.gpx_path = gpx_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.line_color = line_color
        self.line_width = line_width
        self.margin = margin
        self.use_online_map = use_online_map

        self.coords = self.load_gpx()

        self.context = staticmaps.Context()
        self.context.set_tile_provider(staticmaps.tile_provider_OSM)

        gpx_line = staticmaps.Line(
            [staticmaps.create_latlng(lat, lon) for lat, lon in self.coords],
            color=staticmaps.TRANSPARENT,
            width=0
        )
        self.context.add_object(gpx_line)

        center, zoom = self.context.determine_center_zoom(self.width - 2 * self.margin, self.height - 2 * self.margin)

        self.transformer = staticmaps.Transformer(
            self.width,
            self.height,
            zoom,
            center,
            staticmaps.tile_provider_OSM.tile_size()
        )

        self.points = self.project_points()
        self.custom_markers = custom_markers or []
        self.marker_points = self.project_markers()
        self.cached_bg = self.prepare_background()

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
        points = []
        for lat, lon in self.coords:
            latlng = staticmaps.create_latlng(lat, lon)
            x, y = self.transformer.ll2pixel(latlng)
            points.append((int(x), int(y)))
        return points

    def project_markers(self):
        markers = []
        markers.append(("start", self.points[0]))
        for lat, lon, *_ in self.custom_markers:
            closest_idx = 0
            min_d = float('inf')
            for idx, coord in enumerate(self.coords):
                d = self.haversine((lat, lon), coord)
                if d < min_d:
                    min_d = d
                    closest_idx = idx
            markers.append(("custom", self.points[closest_idx]))
        markers.append(("end", self.points[-1]))
        return markers

    def prepare_background(self):
        if not self.use_online_map:
            return None
        pil_image = self.context.render_pillow(self.width, self.height)
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGRA)

    def get_background(self):
        if self.cached_bg is not None:
            return self.cached_bg.copy()
        # Creates a 100% transparent canvas (Alpha channel = 0)
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def render(self):
        frame = self.get_background()

        # Draw the static line
        if len(self.points) > 1:
            cv2.polylines(frame, [np.array(self.points)], False, self.line_color, self.line_width, lineType=cv2.LINE_AA)

        # Draw markers
        for mtype, (x, y) in self.marker_points:
            radius = 12 if mtype in ("start", "end") else 10
            cv2.circle(frame, (x, y), radius, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, (0, 0, 0, 255), 2, lineType=cv2.LINE_AA)

        # --- FIX FOR INSTAGRAM TRANSPARENCY ---
        # 1. Convert from OpenCV standard (BGRA) to Pillow standard (RGBA)
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)

        # 2. Convert array to PIL Image object
        pil_img = Image.fromarray(rgba_frame)

        # 3. Save using Pillow's reliable PNG encoder
        pil_img.save(self.output_path, format="PNG")
        print(f"Instagram-ready PNG saved to → {self.output_path}")

name = "day2"
if __name__ == "__main__":
    renderer = GPXStaticRenderer(
        gpx_path=f"{name}.gpx",
        output_path=f"{name}.png",
        width=1080,
        height=1920,
        line_color=(255, 255, 255, 255),  # Solid white line
        line_width=6,
        margin=150,
        use_online_map=False,  # Must stay False for transparency
        custom_markers=custom_markers,
    )
    renderer.render()