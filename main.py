import sys
import time
import math
import shutil
import numpy as np

# ============================================================
# NOCTURNAL LILY GARDEN
# Vectorized 3D Parametric ASCII + 24-bit ANSI Renderer
# ============================================================

FPS = 30
CHAR_ASPECT = 2.1
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
BLACK = "\033[48;2;0;0;0m"
RESET = "\033[0m"

rng = np.random.default_rng(42)

PALETTES = [
    [(0.00, (110, 20, 95)), (0.28, (240, 65, 155)),
     (0.62, (255, 145, 200)), (1.00, (255, 220, 238))],
    [(0.00, (65, 18, 125)), (0.30, (155, 55, 230)),
     (0.68, (210, 145, 255)), (1.00, (240, 215, 255))],
    [(0.00, (135, 25, 105)), (0.38, (225, 85, 195)),
     (0.72, (255, 145, 210)), (1.00, (255, 195, 230))],
    [(0.00, (90, 25, 100)), (0.35, (190, 65, 175)),
     (0.70, (240, 155, 220)), (1.00, (255, 220, 245))]
]


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def rotate_x(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack((p[..., 0], c*p[..., 1]-s*p[..., 2],
                     s*p[..., 1]+c*p[..., 2]), axis=-1)


def rotate_y(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack((c*p[..., 0]+s*p[..., 2], p[..., 1],
                     -s*p[..., 0]+c*p[..., 2]), axis=-1)


def rotate_z(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack((c*p[..., 0]-s*p[..., 1],
                     s*p[..., 0]+c*p[..., 1], p[..., 2]), axis=-1)


def transform(p, center, rotation):
    p = rotate_x(p, rotation[0])
    p = rotate_y(p, rotation[1])
    p = rotate_z(p, rotation[2])
    return p + np.asarray(center)


class Lily:
    def __init__(self, center, scale, palette, rotation=(0, 0, 0),
                 phase=0.0, bloom=1.0):
        self.center = np.array(center, dtype=float)
        self.scale = scale
        self.palette = PALETTES[palette]
        self.rotation = rotation
        self.phase = phase
        self.bloom = bloom

    def color(self, t):
        result = np.zeros(t.shape + (3,), dtype=float)
        for i in range(len(self.palette)-1):
            x0, c0 = self.palette[i]
            x1, c1 = self.palette[i+1]
            f = np.clip((t-x0)/(x1-x0), 0, 1)
            mask = (t >= x0) & (t <= x1)
            interp = (np.asarray(c0)*(1-f[..., None]) +
                      np.asarray(c1)*f[..., None])
            result[mask] = interp[mask]
        result[t < 0] = self.palette[0][1]
        result[t > 1] = self.palette[-1][1]
        return result

    def petal(self, theta, u, v, t):
        bloom = self.bloom + 0.12*math.sin(t*0.65+self.phase)
        bloom = np.clip(bloom, 0.12, 1.12)
        spread = 0.30 + 1.12*bloom
        radial = spread*u
        width = 0.31*np.sin(np.pi*u)**0.75*(1-0.60*u)
        ridge = 0.105*np.sin(np.pi*u)
        curl = 0.30*np.maximum(0, u-0.68)**2*5
        height = (ridge*(1-v*v) -
                  0.13*v*v*np.sin(np.pi*u) + curl)
        height += 0.035*np.sin(3*np.pi*u+theta)*v
        x = radial*np.cos(theta)-width*v*np.sin(theta)
        y = radial*np.sin(theta)+width*v*np.cos(theta)
        z = height+0.07*np.sin(2*np.pi*u)
        local = np.stack((x, y, z), axis=-1)
        return transform(local*self.scale, self.center, self.rotation)

    def geometry(self, t, resolution=100):
        u = np.linspace(0.008, 0.995, resolution)
        v = np.linspace(-0.99, 0.99, resolution//2)
        U, V = np.meshgrid(u, v, indexing="ij")
        for tier in range(2):
            for i in range(3):
                angle = (2*np.pi*i/3 + tier*np.pi/3 +
                         0.08*math.sin(t*0.4+self.phase))
                yield self.petal(angle, U, V, t), U, V, tier


def stem_surface(start, end, width, t, resolution=100):
    u = np.linspace(0, 1, resolution)
    v = np.linspace(-1, 1, 7)
    U, V = np.meshgrid(u, v, indexing="ij")
    start, end = np.asarray(start), np.asarray(end)
    P = start[None, None, :]*(1-U[..., None])
    P += end[None, None, :]*U[..., None]
    P[..., 0] += 0.09*np.sin(U*np.pi+t)*U
    P[..., 2] += 0.035*np.sin(U*np.pi)
    P[..., 1] += V*width
    return P


def leaf_surface(start, direction, length, width, t):
    u = np.linspace(0.01, 0.99, 90)
    v = np.linspace(-1, 1, 13)
    U, V = np.meshgrid(u, v, indexing="ij")
    direction = normalize(np.asarray(direction))
    side = normalize(np.cross(direction, np.array([0., 0., 1.])))
    W = width*np.sin(np.pi*U)**0.85
    P = (np.asarray(start) + direction*(U*length)[..., None] +
         side*(V*W)[..., None])
    P[..., 2] += 0.12*np.sin(np.pi*U)*(1-V**2)
    return P


# ============================================================
# CAMERA - CORRECTED
# ============================================================

CAMERA = np.array([0.0, -0.25, 5.0])

def project(points, width, height):
    z = points[..., 2]

    focal = min(width, height * CHAR_ASPECT) * 1.45
    depth = np.maximum(5.0 - z, 0.5)

    px = width / 2 + focal * points[..., 0] / depth

    py = (
        height * 0.43
        - focal * points[..., 1] / depth / CHAR_ASPECT
    )

    return px, py, z


def project(points, width, height):
    relative = points-CAMERA
    z = relative[..., 2]
    focal = min(width, height*CHAR_ASPECT)*0.85
    safe_z = np.maximum(z, -5.0)
    px = width/2 + focal*relative[..., 0]/(8.0-safe_z)
    py = height/2 - (focal*relative[..., 1]/(8.0-safe_z))/CHAR_ASPECT
    return px, py, safe_z


class Renderer:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.z_buffer = np.full((height, width), -np.inf)
        self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.chars = np.full((height, width), " ", dtype="<U1")

    def clear(self):
        self.z_buffer.fill(-np.inf)
        self.rgb.fill(0)
        self.chars.fill(" ")

    def rasterize(self, points, base_color, brightness=1.0, freckles=False):
        x, y, z = project(points, self.width, self.height)
        du = (np.gradient(points, axis=0) if points.shape[0] > 1
              else np.zeros_like(points))
        dv = (np.gradient(points, axis=1) if points.shape[1] > 1
              else np.zeros_like(points))
        raw_normals = np.cross(du, dv)
        lengths = np.linalg.norm(raw_normals, axis=-1, keepdims=True)
        normals = np.where(lengths > 1e-9,
                           raw_normals/np.maximum(lengths, 1e-9),
                           np.array([0.0, 0.0, 1.0]))
        light = normalize(np.array([-0.35, -0.55, 0.95]))
        diffuse = np.maximum(np.sum(normals*light, axis=-1), 0)
        rim = np.power(1-np.abs(normals[..., 2]), 2)
        illumination = (0.28+0.65*diffuse+0.24*rim)*brightness
        rgb = np.clip(base_color*illumination[..., None], 0, 255).astype(np.uint8)

        if freckles:
            throat = (points[..., 0]**2+points[..., 1]**2) < 0.35
            pattern = np.sin(points[..., 0]*105)*np.sin(points[..., 1]*115)
            dots = throat & (pattern > 0.72)
            rgb[dots] = (np.array([85, 12, 48])*brightness).astype(np.uint8)

        ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
        valid = ((ix >= 0) & (ix < self.width) &
                 (iy >= 0) & (iy < self.height) & np.isfinite(z))
        ix, iy, zz, colors = ix[valid], iy[valid], z[valid], rgb[valid]
        visible = zz > self.z_buffer[iy, ix]
        ix, iy, zz, colors = ix[visible], iy[visible], zz[visible], colors[visible]
        self.z_buffer[iy, ix] = zz
        self.rgb[iy, ix] = colors
        luminance = (0.2126*colors[:, 0]+0.7152*colors[:, 1]+
                     0.0722*colors[:, 2])/255
        ramp = np.array(list(RAMP))
        indices = np.clip((luminance*(len(RAMP)-1)).astype(int), 0, len(RAMP)-1)
        self.chars[iy, ix] = ramp[indices]

    def render_stamens(self, lily, t, brightness):
        for i in range(6):
            angle = 2*np.pi*i/6
            u = np.linspace(0, 1, 45)
            radius = (0.10+0.38*u)*lily.scale
            height = (0.08+0.23*u+0.09*np.sin(np.pi*u))*lily.scale
            P = np.stack((radius*np.cos(angle), radius*np.sin(angle),
                          height), axis=-1)
            P = transform(P, lily.center, lily.rotation)[:, None, :]
            color = np.full(P.shape[:-1]+(3,), [255, 190, 70], dtype=float)
            self.rasterize(P, color, brightness*1.25)
            tip = transform(np.array([[
                0.48*lily.scale*np.cos(angle),
                0.48*lily.scale*np.sin(angle),
                0.33*lily.scale]]), lily.center, lily.rotation)
            self.rasterize(tip[:, None, :],
                           np.full((1, 1, 3), [255, 160, 55]),
                           brightness*1.3)

    def render_lily(self, lily, t, brightness):
        for P, U, V, tier in lily.geometry(t):
            gradient = np.clip(U**0.85+0.04*tier, 0, 1)
            color = lily.color(gradient)
            vein = np.exp(-V**2*65)
            color = np.clip(color*(0.92+0.13*vein[..., None]), 0, 255)
            if tier:
                color *= 0.93
            self.rasterize(P, color, brightness, freckles=(tier == 0))
        self.render_stamens(lily, t, brightness)


def create_bouquet():
    return [
        Lily(
            (-0.15, 0.15, 0.35),
            1.15, 0,
            rotation=(0.05, -0.08, 0.10),
            phase=0.0,
            bloom=1.0
        ),

        Lily(
            (-1.50, -0.40, -0.45),
            0.80, 1,
            rotation=(0.15, 0.15, -0.25),
            phase=1.7,
            bloom=0.95
        ),

        Lily(
            (1.45, -0.45, -0.50),
            0.80, 2,
            rotation=(0.12, -0.15, 0.30),
            phase=3.2,
            bloom=0.95
        ),

        Lily(
            (0.85, 1.15, -0.65),
            0.55, 3,
            rotation=(0.10, 0.10, 0.20),
            phase=4.2,
            bloom=0.75
        )
    ]


class Stardust:
    def __init__(self, width, height):
        self.width, self.height = width, height
        colors = [(255, 180, 215), (205, 165, 255), (255, 210, 135)]
        self.hero = [
            (rng.uniform(0, width), rng.uniform(0, height),
             rng.uniform(0, 2*np.pi), colors[i % 3], rng.uniform(0.5, 1.5))
            for i in range(15)
        ]
        self.micro = [
            (rng.uniform(0, width), rng.uniform(0, height),
             rng.uniform(0, 2*np.pi), rng.choice([".", "+", "·"]))
            for _ in range(120)
        ]

    def resize(self, width, height):
        if width != self.width or height != self.height:
            self.__init__(width, height)

    def render(self, renderer, t, brightness):
        h, w = renderer.height, renderer.width
        for x, y, phase, color, speed in self.hero:
            pulse = 0.45+0.55*math.sin(t*speed+phase)**2
            if pulse < 0.28:
                continue
            ix, iy = int(x), int(y)
            pattern = [
                (0, 0, "*", 1.0), (-2, 0, ".", 0.5),
                (-1, 0, "-", 0.8), (1, 0, "-", 0.8),
                (2, 0, ".", 0.5), (0, -2, ".", 0.5),
                (0, -1, "|", 0.8), (0, 1, "|", 0.8),
                (0, 2, ".", 0.5)
            ]
            for dx, dy, char, factor in pattern:
                xx, yy = ix+dx, iy+dy
                if (0 <= xx < w and 0 <= yy < h and
                        not np.isfinite(renderer.z_buffer[yy, xx])):
                    c = np.clip(np.array(color)*pulse*factor*brightness,
                                0, 255).astype(np.uint8)
                    renderer.rgb[yy, xx] = c
                    renderer.chars[yy, xx] = char

        for x, y, phase, char in self.micro:
            ix, iy = int(x), int(y)
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            if np.isfinite(renderer.z_buffer[iy, ix]):
                continue
            pulse = 0.2+0.8*(0.5+0.5*math.sin(t*2.2+phase))
            c = int(110*pulse*brightness)
            renderer.rgb[iy, ix] = (int(c*0.85), int(c*0.78), c)
            renderer.chars[iy, ix] = char


def render_botany(renderer, t, brightness):
    green = np.array([55, 125, 78], dtype=float)
    starts = [(-0.1, -0.85, -0.3), (-1.65, -1.0, -0.8),
              (1.5, -1.0, -1.0), (0.95, 0.4, -1.0)]
    ends = [(-0.05, 0.05, 0.25), (-1.65, -0.30, -0.65),
            (1.5, -0.42, -0.90), (0.95, 1.05, -1.15)]

    for index, (start, end) in enumerate(zip(starts, ends)):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        end = end+np.array([0.08*math.sin(t*0.8+index), 0, 0])
        stem = stem_surface(start, end, 0.035, t)
        color = np.broadcast_to(green, stem.shape[:-1]+(3,)).copy()
        renderer.rasterize(stem, color, brightness*0.8)

        direction = end-start
        for j in range(3):
            u = 0.25+j*0.22
            leaf_start = start*(1-u)+end*u
            sign = -1 if j % 2 else 1
            leaf_direction = np.array([sign*(0.8+0.1*j), 0.25, 0.5])
            leaf_direction += direction*0.2
            leaf = leaf_surface(leaf_start, leaf_direction,
                                0.72-0.08*j, 0.095, t)
            leaf_color = np.broadcast_to(
                [30+12*j, 95+15*j, 60+8*j],
                leaf.shape[:-1]+(3,)).copy()
            renderer.rasterize(leaf, leaf_color, brightness*0.85)


def make_frame(renderer):
    output = [BLACK]
    previous_color = None
    for y in range(renderer.height):
        for x in range(renderer.width):
            color = tuple(int(v) for v in renderer.rgb[y, x])
            char = renderer.chars[y, x]
            if char == " ":
                color = (0, 0, 0)
            if color != previous_color:
                output.append(f"\033[38;2;{color[0]};{color[1]};{color[2]}m")
                previous_color = color
            output.append(char)
        output.append("\033[0m\n")
        output.append(BLACK)
        previous_color = None
    output.append(RESET)
    return "".join(output)


def main():
    sys.stdout.write("\033[?1049h\033[?25l\033[2J\033[H")
    sys.stdout.flush()
    start_time = time.perf_counter()
    lilies = create_bouquet()
    stars = renderer = None

    try:
        while True:
            frame_start = time.perf_counter()
            terminal = shutil.get_terminal_size(fallback=(100, 40))
            width = max(40, terminal.columns)
            height = max(16, terminal.lines-1)

            if (renderer is None or renderer.width != width or
                    renderer.height != height):
                renderer = Renderer(width, height)
                stars = Stardust(width, height)

            renderer.clear()
            elapsed = time.perf_counter()-start_time
            fade = np.clip(elapsed/1.5, 0, 1)
            fade = fade*fade*(3-2*fade)
            t = elapsed
            breeze = 0.035*math.sin(t*0.75)

            for i, lily in enumerate(lilies):
                original_rotation = lily.rotation
                lily.rotation = (
                    original_rotation[0]+breeze,
                    original_rotation[1]+0.035*math.sin(t*0.55+i),
                    original_rotation[2]+0.025*math.sin(t*0.7+i*1.3)
                )
                renderer.render_lily(lily, t, fade)
                lily.rotation = original_rotation

            render_botany(renderer, t, fade)
            stars.resize(width, height)
            stars.render(renderer, t, fade)
            sys.stdout.write("\033[H"+make_frame(renderer))
            sys.stdout.flush()

            duration = time.perf_counter()-frame_start
            time.sleep(max(0, 1/FPS-duration))

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[0m\033[?25h\033[?1049l")
        sys.stdout.flush()


if __name__ == "__main__":
    main()