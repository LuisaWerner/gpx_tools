import xml.etree.ElementTree as ET
import cv2
import numpy as np
import staticmaps
from PIL import Image

custom_markers = [
    (45.432, 6.380, "Col de la Madeleine")
]


class GPXStaticRenderer:
    """
    GPX → Ultra-High-Resolution Transparent Static PNG overlay optimized for production edits.
    Uses sub-pixel fixed-point rendering to eliminate jagged edges and maximize anti-aliasing quality.
    """

    def __init__(
            self,
            gpx_path,
            output_path="mini_marmotte_overlay.png",
            width=1080,
            height=1920,
            scale=4,  # Bumped to 4x scale (4320 x 7680 Ultra-HD canvas)
            line_color=(255, 255, 255, 255),  # BGRA (White)
            line_width=6,  # Base width, scales automatically
            margin=100,
            use_online_map=False,
            custom_markers=None
    ):
        self.gpx_path = gpx_path
        self.output_path = output_path

        # Apply the scaling factor to dimensions and margins
        self.width = width * scale
        self.height = height * scale
        self.line_width = line_width * scale
        self.margin = margin * scale

        self.line_color = line_color
        self.use_online_map = use_online_map

        # Sub-pixel rendering configurations (2^4 = 16 positions per pixel for hyper-smooth curves)
        self.shift = 4
        self.sub_scale = 1 << self.shift

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

        # Base marker configurations (unscaled for sub-pixel layer handling)
        self.marker_radius_large = 12 * scale
        self.marker_radius_small = 10 * scale
        self.marker_border_thickness = max(2, 1 * scale)

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
        """
        Projects map coordinates directly into scaled up sub-pixel precision integers.
        """
        points = []
        for lat, lon in self.coords:
            latlng = staticmaps.create_latlng(lat, lon)
            x, y = self.transformer.ll2pixel(latlng)
            # Scale coordinates up by fixed point factor
            sub_x = int(round(x * self.sub_scale))
            sub_y = int(round(y * self.sub_scale))
            points.append((sub_x, sub_y))
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
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def render(self):
        frame = self.get_background()

        # Render paths using fixed-point subpixel positions for flawlessly smooth vector curves
        if len(self.points) > 1:
            cv2.polylines(
                frame,
                [np.array(self.points, dtype=np.int32)],
                False,
                self.line_color,
                self.line_width,
                lineType=cv2.LINE_AA,
                shift=self.shift
            )

        # Render markers using fixed-point radius adjustments
        for mtype, (x, y) in self.marker_points:
            radius = self.marker_radius_large if mtype in ("start", "end") else self.marker_radius_small
            sub_radius = int(round(radius * self.sub_scale))

            # Filled White Solid Core
            cv2.circle(frame, (x, y), sub_radius, (255, 255, 255, 255), -1, lineType=cv2.LINE_AA, shift=self.shift)
            # Crisp Black Outline Stroke
            cv2.circle(
                frame,
                (x, y),
                sub_radius,
                (0, 0, 0, 255),
                self.marker_border_thickness,
                lineType=cv2.LINE_AA,
                shift=self.shift
            )

        # Convert OpenCV (BGRA) matrix over to Pillow (RGBA) format
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba_frame)

        # Save with full PNG bit depth optimizations enabled
        pil_img.save(self.output_path, format="PNG", optimize=True, compress_level=9)
        print(f"Master-quality export saved to → {self.output_path} ({self.width}x{self.height} px)")


name = "rhoen"
if __name__ == "__main__":
    renderer = GPXStaticRenderer(
        gpx_path=f"{name}.gpx",
        output_path=f"{name}.png",
        width=1080,
        height=1920,
        scale=4,  # Change to 5 or 6 if you need absolutely extreme canvas sizing
        line_color=(255, 255, 255, 255),
        line_width=5,
        margin=150,
        use_online_map=False,
        custom_markers=custom_markers,
    )
    renderer.render()