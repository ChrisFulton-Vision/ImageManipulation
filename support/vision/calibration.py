import copy
import os
import pickle as pkl
import re
from datetime import datetime
from os.path import join
import sys, types
import numpy as np
from support.runtime.pixel_handler import Pixel as pxl
from support.mathHelpers.include_numba import _njit as njit, prange

class Calibration:
    def __init__(self, filepath=None):
        # cache bits first so __setattr__ can use them safely
        self._validCal_dirty = True
        self._validCal_cache = False
        self._VALID_FIELDS = ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'width', 'height']

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.k1 = None
        self.k2 = None
        self.p1 = None
        self.p2 = None
        self.k3 = None
        self.k4 = None
        self.calTime = None
        self.numCBUsed = None
        self.rmsError = None
        self.width = None
        self.height = None
        self.hfov = None
        self.has_tangential = False
        self.fisheye = False
        self.calDatetime = None

        self.remapK = None

        self.cov = np.eye(4).astype(dtype=np.float32) * 10.0

        self.scale = 1.0

        if filepath is not None:
            self.fromFile(filepath)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in getattr(self, "_VALID_FIELDS", ()):
            # any change to these means validity might have changed
            object.__setattr__(self, "_validCal_dirty", True)

    def __str__(self):
        return self.calStr

    @property
    def iteratable_params(self):
        if not self.fisheye:
            return self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2, self.k3
        else:
            return self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.k3, self.k4

    def randomize(
            self,
            rng=None,
            *,
            # camera geometry
            width=None,
            height=None,
            keep_intrinsics=True,
            fx_range=(600.0, 4500.0),
            fy_range=None,  # if None -> fy ~ fx * (1+small jitter)
            principal_jitter_px=5.0,  # if keep_intrinsics=False; cx/cy jitter about center
            # distortion regime controls
            profile="mixed",  # "radial_only", "tangential_only", "mixed"
            strength="medium",  # "low", "medium", "high"
            # safety / validity
            ensure_invertible=True,
            min_abs_L=0.20,  # enforce |L(r)| >= this on [0, r_max]
            max_attempts=200
    ):
        """
        Randomize calibration parameters for benchmarking.

        - profile:
            "radial_only"      -> p1=p2=0, random k1,k2,k3
            "tangential_only"  -> k1=k2=k3=0, random p1,p2
            "mixed"            -> random both
        - strength: controls typical magnitude ranges.
        - ensure_invertible: rejects samples where radial scale L(r) gets too small
          over the image FOV (helps avoid pathological inversions).

        Returns: self (mutates in place).
        """
        if rng is None:
            rng = np.random.default_rng()

        # -----------------------
        # Decide resolution
        # -----------------------
        if width is None:
            width = self.width if self.width is not None else 864
        if height is None:
            height = self.height if self.height is not None else 864
        self.width = int(width)
        self.height = int(height)

        # -----------------------
        # Intrinsics
        # -----------------------
        if not keep_intrinsics or (self.fx is None or self.fy is None or self.cx is None or self.cy is None):
            fx_lo, fx_hi = fx_range
            self.fx = float(rng.uniform(fx_lo, fx_hi))

            if fy_range is None:
                # keep near-square pixels but not exact
                self.fy = float(self.fx * rng.uniform(0.97, 1.03))
            else:
                fy_lo, fy_hi = fy_range
                self.fy = float(rng.uniform(fy_lo, fy_hi))

            # principal point near image center with mild jitter
            self.cx = float((self.width - 1) * 0.5 + rng.uniform(-principal_jitter_px, principal_jitter_px))
            self.cy = float((self.height - 1) * 0.5 + rng.uniform(-principal_jitter_px, principal_jitter_px))

        # -----------------------
        # Distortion magnitude presets (Brown–Conrady)
        # These are *practical* ranges; adjust as you learn what you want to stress.
        # -----------------------
        if strength == "low":
            k1_rng = (-0.05, 0.05)
            k2_rng = (-0.05, 0.05)
            k3_rng = (-0.02, 0.02)
            p_rng = (-5e-4, 5e-4)
        elif strength == "medium":
            k1_rng = (-0.20, 0.20)
            k2_rng = (-0.20, 0.20)
            k3_rng = (-0.10, 0.10)
            p_rng = (-2e-3, 2e-3)
        else:  # "high"
            k1_rng = (-0.80, 0.80)
            k2_rng = (-0.80, 0.80)
            k3_rng = (-0.60, 0.60)
            p_rng = (-1e-2, 1e-2)

        # We'll sample until we pass safety checks (or give up).
        # Use r_max at the image corner in normalized coords.
        def r_max_norm():
            # corners relative to principal point, normalized by focal length
            xs = np.array([0.0, self.width - 1.0])
            ys = np.array([0.0, self.height - 1.0])
            X, Y = np.meshgrid(xs, ys)
            xn = (X.ravel() - self.cx) / self.fx
            yn = (Y.ravel() - self.cy) / self.fy
            return float(np.sqrt((xn * xn + yn * yn).max()))

        rmax = r_max_norm()

        def min_abs_L_over_fov(k1, k2, k3):
            # sample L(r) over [0, rmax] to avoid L near 0 in-view
            rs = np.linspace(0.0, rmax, 64, dtype=np.float64)
            r2 = rs * rs
            L = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            return float(np.min(np.abs(L)))

        for _ in range(max_attempts):
            if profile == "tangential_only":
                k1 = k2 = k3 = 0.0
                p1 = float(rng.uniform(*p_rng))
                p2 = float(rng.uniform(*p_rng))
            elif profile == "radial_only":
                p1 = p2 = 0.0
                k1 = float(rng.uniform(*k1_rng))
                k2 = float(rng.uniform(*k2_rng))
                k3 = float(rng.uniform(*k3_rng))
            else:  # "mixed"
                k1 = float(rng.uniform(*k1_rng))
                k2 = float(rng.uniform(*k2_rng))
                k3 = float(rng.uniform(*k3_rng))
                p1 = float(rng.uniform(*p_rng))
                p2 = float(rng.uniform(*p_rng))

            if ensure_invertible:
                if min_abs_L_over_fov(k1, k2, k3) < float(min_abs_L):
                    continue

            # accept
            self.k1 = k1
            self.k2 = k2
            self.k3 = k3
            self.p1 = p1
            self.p2 = p2
            break
        else:
            raise RuntimeError("Failed to sample a stable calibration (increase max_attempts or relax min_abs_L).")

        # Accessories (keep validCal happy)
        self.fisheye = False
        self.calTime = 0.0
        self.numCBUsed = 0
        self.rmsError = 0.0
        # hfov is optional; leave if already present, otherwise approximate from fx
        if self.hfov is None:
            # approximate horizontal FOV in degrees
            self.hfov = float(2.0 * np.degrees(np.arctan((self.width * 0.5) / self.fx)))

        # validCal cache will be marked dirty automatically via __setattr__
        return self

    def setCameraMatrix(self, mtx=None, fx=None, fy=None, cx=None, cy=None):
        if mtx is not None:
            self.fx = mtx[0, 0]
            self.fy = mtx[1, 1]
            self.cx = mtx[0, 2]
            self.cy = mtx[1, 2]
        elif fx is not None and fy is not None and cx is not None and cy is not None:
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
        else:
            self.fx = None
            self.fy = None
            self.cx = None
            self.cy = None
        self.scale = 1.0

    @property
    def K(self):
        return self.getCameraMatrix()

    @property
    def inv(self):
        if not hasattr(self, "scale"):
            self.scale = 1.0

        if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None and self.scale is not None:
            fx = self.scale * self.fx
            fy = self.scale * self.fy
            cx = self.scale * (self.cx + 0.5) - 0.5
            cy = self.scale * (self.cy + 0.5) - 0.5
            return np.array([[1.0 / fx, 0.0, -cx / fx],
                             [0.0, 1.0 / fy, -cy / fy],
                             [0.0, 0.0, 1.0]])

        return None

    def getCameraMatrix(self):
        # Included for backwards compatibility
        if not hasattr(self, "scale"):
            self.scale = 1.0

        if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None and self.scale is not None:
            fx = self.scale * self.fx
            fy = self.scale * self.fy
            cx = self.scale * (self.cx + 0.5) - 0.5
            cy = self.scale * (self.cy + 0.5) - 0.5
            return np.array([[fx, 0.0, cx],
                             [0.0, fy, cy],
                             [0.0, 0.0, 1.0]])

        return None

    def setDistortion(self, dist=None, k1=None, k2=None, p1=None, p2=None, k3=None):
        if dist is not None and self.fisheye:
            self.k1 = dist[0]
            self.k2 = dist[1]
            self.k3 = dist[2]
            self.k4 = dist[3]
        elif dist is not None:
            self.k1 = dist[0]
            self.k2 = dist[1]
            self.p1 = dist[2]
            self.p2 = dist[3]
            self.k3 = dist[4]
        elif k1 is not None and k2 is not None and p1 is not None and p2 is not None and k3 is not None:
            self.k1 = k1
            self.k2 = k2
            self.p1 = p1
            self.p2 = p2
            self.k3 = k3
        else:
            self.k1 = None
            self.k2 = None
            self.p1 = None
            self.p2 = None
            self.k3 = None
            self.k4 = None

    def getDistortion(self):
        if self.fisheye:
            if self.k1 is not None and self.k2 is not None and self.k3 is not None and self.k4 is not None:
                return np.array([self.k1, self.k2, self.k3, self.k4]).flatten()
            else:
                return None

        if self.k1 is not None and self.k2 is not None and self.p1 is not None and self.p2 is not None and self.k3 is not None:
            return np.array([self.k1, self.k2, self.p1, self.p2, self.k3]).flatten()
        else:
            return None

    def setAccessories(self, calTime, numCBUsed, width, height, hfov, rms, timeOfCompute=None):

        self.calTime = calTime
        self.numCBUsed = numCBUsed
        self.width = width
        self.height = height
        self.hfov = hfov
        self.rmsError = rms
        self.calDatetime = timeOfCompute

    @property
    def calStr(self):
        mtx = self.getCameraMatrix()
        dist = self.getDistortion()

        calStr = ''
        if self.validCal:
            calStr += '# Camera matrix'
            if self.calDatetime is not None:
                calStr += ' computed at:\n#{}\n'.format(datetime.date(self.calDatetime))
            calStr += '# Original size: ' + str(int(self.width)) + 'x' + str(int(self.height)) + '\n'
            if self.fisheye:
                calStr += '#Fisheye Cal'
            calStr += '\nfx={:.{}f}'.format(mtx[0, 0], 10) + '\n'
            calStr += 'fy={:.{}f}'.format(mtx[1, 1], 10) + '\n'
            calStr += 'cx={:.{}f}'.format(mtx[0, 2], 10) + '\n'
            calStr += 'cy={:.{}f}'.format(mtx[1, 2], 10) + '\n\n'

            calStr += '#Distortion coefficients\n'
            if self.fisheye:
                calStr += 'k1={:.{}f}'.format(dist[0], 10) + '\n'
                calStr += 'k2={:.{}f}'.format(dist[1], 10) + '\n'
                calStr += 'k3={:.{}f}'.format(dist[2], 10) + '\n'
                calStr += 'k4={:.{}f}'.format(dist[3], 10) + '\n\n'
            else:
                calStr += 'k1={:.{}f}'.format(dist[0], 10) + '\n'
                calStr += 'k2={:.{}f}'.format(dist[1], 10) + '\n'
                calStr += 'p1={:.{}f}'.format(dist[2], 10) + '\n'
                calStr += 'p2={:.{}f}'.format(dist[3], 10) + '\n'
                calStr += 'k3={:.{}f}'.format(dist[4], 10) + '\n\n'

            calStr += '#Total cal time (sec)\n'
            calStr += 'ct={:.{}f}'.format(self.calTime, 10) + '\n\n'

            calStr += '#Chessboards used\n'
            calStr += 'total=' + str(self.numCBUsed) + '\n'
            calStr += 'valid=' + str(self.numCBUsed) + '\n'
            calStr += 'rmsErr=' + str(self.rmsError) + '\n\n'

            calStr += '#Other\n'
            calStr += 'resolution=' + str(int(self.scale * self.width)) + 'x' + str(
                int(self.scale * self.height)) + '\n'
            calStr += 'hfov=' + str(self.hfov) + "\n"
        else:
            calStr = 'Bad Cal'

        return calStr

    def copy(self):
        return copy.deepcopy(self)

    def copy_from(self, calToCopy):
        self.__init__()
        for obj in calToCopy.__dict__:
            try:
                self.__dict__[obj] = calToCopy.__dict__[obj]
            except KeyError:
                # Allows for versioning issues, changed naming conventions.
                print("Older version...")
                self.calDatetime = None

    def toBinFile(self, fileDirectory):
        with open(join(fileDirectory, 'calibration.pkl'), 'wb') as file:
            pkl.dump(self, file)

    def fromBinFile(self, fileDirectory):

        # Resolve the .pkl path (supports either a directory or direct .pkl file)
        if os.path.isdir(fileDirectory):
            filepath = join(fileDirectory, 'calibration.pkl')
        else:
            filepath = fileDirectory

        if not (os.path.exists(filepath) and filepath.lower().endswith('.pkl')):
            return False

        class _CompatUnpickler(pkl.Unpickler):
            def find_class(self, module, name):
                # Old location recorded in the pickle
                if module in ('Calibration', '__main__') and name == 'Calibration':
                    return Calibration  # current class in this module
                # New location (if you ever rename/move again, add more mappings here)
                if module.endswith('SupportModules.Calibration') and name == 'Calibration':
                    return Calibration
                return super().find_class(module, name)

        with open(filepath, 'rb') as f:
            try:
                obj = _CompatUnpickler(f).load()
            except ModuleNotFoundError:
                # Fallback: quickly shim a fake module named "Calibration" that points here
                shim = types.ModuleType('Calibration')
                shim.Calibration = Calibration
                sys.modules.setdefault('Calibration', shim)
                f.seek(0)
                obj = pkl.load(f)

        self.copy_from(obj)
        return True

    def toFile(self, fileDirectory):
        with open(join(fileDirectory, 'calibration.txt'), 'w') as file:
            file.write(self.calStr)

    def fromFile(self, fileDirectory):
        if os.path.exists(join(fileDirectory, 'calibration.txt')):
            filepath = join(fileDirectory, 'calibration.txt')
        elif os.path.exists(fileDirectory) and fileDirectory.lower().endswith('.txt'):
            filepath = fileDirectory
        else:
            return False

        self.__init__()

        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                for line in file:
                    match line[0:2]:
                        case '#2':
                            self.calDatetime = datetime.strptime(line[1:-1], '%Y-%m-%d')
                        case '#F':
                            self.fisheye = True
                        case 'fx':
                            self.fx = float(line[3:])
                        case 'fy':
                            self.fy = float(line[3:])
                        case 'cx':
                            self.cx = float(line[3:])
                        case 'cy':
                            self.cy = float(line[3:])
                        case 'k1':
                            self.k1 = float(line[3:])
                        case 'k2':
                            self.k2 = float(line[3:])
                        case 'k3':
                            self.k3 = float(line[3:])
                        case 'k4':
                            self.k4 = float(line[3:])
                        case 'p1':
                            self.p1 = float(line[3:])
                        case 'p2':
                            self.p2 = float(line[3:])
                        case 'ct':
                            self.calTime = float(line[3:])
                        case 'to':
                            self.numCBUsed = float(line[6:])
                        case 'rm':
                            self.rmsError = float(line[7:])
                        case 're':
                            intList = re.findall(r'\d+', line)
                            self.width = int(intList[0])
                            self.height = int(intList[1])
                        case 'hf':
                            self.hfov = float(line[7:])
        if self.validCal:
            return True
        else:
            return False

    def _compute_validCal(self) -> bool:
        # (this is your existing logic, just moved into a function)
        if self.fisheye and any([
            self.fx is None, self.fy is None, self.cx is None, self.cy is None,
            self.k1 is None, self.k2 is None, self.k3 is None, self.k4 is None,
            self.calTime is None, self.numCBUsed is None, self.rmsError is None,
            self.width is None, self.height is None, self.hfov is None
        ]):
            return False
        elif not self.fisheye and any([
            self.fx is None, self.fy is None, self.cx is None, self.cy is None,
            self.k1 is None, self.k2 is None, self.p1 is None, self.p2 is None, self.k3 is None,
            self.calTime is None, self.numCBUsed is None, self.rmsError is None,
            self.width is None, self.height is None, self.hfov is None
        ]):
            return False
        self.has_tangential = self.p1 != 0.0 or self.p2 != 0.0
        return True

    @property
    def validCal(self) -> bool:
        '''
        >>> cal = default_864_cam()
        >>> assert cal.validCal
        >>> cal2 = default_2848_cam()
        >>> assert cal2.validCal
        '''
        try:
            self.__getattribute__('_validCal_dirty')
        except AttributeError:
            self._validCal_dirty = True

        if self._validCal_dirty:
            self._validCal_cache = self._compute_validCal()
            self._validCal_dirty = False
        return self._validCal_cache

    def scaleCalibration(self, newWidth: int):

        if not self.validCal:
            raise ValueError('Invalid Calibration. Missing necessary parameter.')

        self.scale = newWidth / self.width

    def havePix_needNorm(self, pixel: pxl):
        '''
        >>> cal = default_864_cam()
        >>> p = pxl(pix_coords=(cal.cx, cal.cy))
        >>> cal.havePix_needNorm(p)
        >>> assert round(p.norm_coords[0], 12) == 0.0
        >>> assert  round(p.norm_coords[1], 12) == 0.0

        >>> cal = default_2848_cam()
        >>> p = pxl(pix_coords=(cal.cx, cal.cy))
        >>> cal.havePix_needNorm(p)
        >>> assert round(p.norm_coords[0], 12) == 0.0
        >>> assert  round(p.norm_coords[1], 12) == 0.0
        '''
        if not self.validCal:
            raise ValueError("Calibration invalid.")
        if pixel.pix_coords is not None:
            pixel.norm_coords = [(pixel.pix_coords[0] - self.cx) / self.fx,
                                 (pixel.pix_coords[1] - self.cy) / self.fy]
            return
        raise ValueError("Pix_coords must exist before calling this function.")

    def haveNorm_needPix(self, pixel: pxl):
        '''
        >>> cal = default_864_cam()
        >>> p = pxl(pix_coords=(cal.cx, cal.cy))
        >>> cal.havePix_needNorm(p)
        >>> assert round(p.norm_coords[0], 12) == 0.0
        >>> assert  round(p.norm_coords[1], 12) == 0.0
        '''
        if not self.validCal:
            raise ValueError("Calibration invalid.")
        if pixel.norm_coords is not None:
            pixel.pix_coords = [pixel.norm_coords[0] * self.fx + self.cx,
                                pixel.norm_coords[1] * self.fy + self.cy]
            return
        raise ValueError("Norm_coords must exist before calling this function.")

    def max_corner_distortion_px(self, corners_px_und: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        """Return the maximum *forward* distortion magnitude (in pixels) at image corners.

        This is a simple "calibration nastiness" proxy: how far the forward model
        moves ideal (undistorted) corner pixels when mapped into the distorted image.

        Parameters
        ----------
        corners_px_und : np.ndarray | None
            Optional array of undistorted pixel points, shape (N,2).
            If None, uses the four image corners:
            (0,0), (W-1,0), (W-1,H-1), (0,H-1).

        Returns
        -------
        max_mag_px : float
            Maximum Euclidean displacement magnitude in pixels.
        mags_px : np.ndarray
            Per-point displacement magnitudes in pixels (shape (N,)).
        """
        if not self.validCal:
            raise ValueError("Calibration invalid.")
        if self.width is None or self.height is None:
            raise ValueError("Calibration missing width/height.")

        if corners_px_und is None:
            w = int(self.width)
            h = int(self.height)
            corners_px_und = np.array(
                [[0.0, 0.0],
                 [float(w - 1), 0.0],
                 [float(w - 1), float(h - 1)],
                 [0.0, float(h - 1)]],
                dtype=np.float64,
            )
        else:
            corners_px_und = np.asarray(corners_px_und, dtype=np.float64)
            if corners_px_und.ndim != 2 or corners_px_und.shape[1] != 2:
                raise ValueError(f"corners_px_und must have shape (N,2); got {corners_px_und.shape}")

        corners_px_dist = distort_points_px(self, corners_px_und)
        d = corners_px_dist - corners_px_und
        mags = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
        return float(np.max(mags)), mags

    def orientation_preservation_metrics(self, *, grid=25, eps_px=1.0):
        """
        Numerical Jacobian diagnostics for the forward distortion map in PIXEL SPACE.

        Returns:
          min_detJ
          frac_neg_detJ
          min_abs_detJ
          max_kappa_proxy   (conditioning proxy; larger is worse)
        """
        assert self.validCal

        xs = np.linspace(0.0, self.width - 1.0, grid, dtype=np.float64)
        ys = np.linspace(0.0, self.height - 1.0, grid, dtype=np.float64)

        dets = []
        kappas = []

        for y in ys:
            for x in xs:
                p = np.array([[x, y]], dtype=np.float64)
                px = np.array([[x + eps_px, y]], dtype=np.float64)
                py = np.array([[x, y + eps_px]], dtype=np.float64)

                do = distort_points_px(self, p)[0]
                dx = distort_points_px(self, px)[0]
                dy = distort_points_px(self, py)[0]

                jx = (dx - do) / eps_px  # column for +x
                jy = (dy - do) / eps_px  # column for +y

                # J = [jx jy] with jx,jy as 2-vectors
                a, c = jx[0], jx[1]
                b, d = jy[0], jy[1]

                detJ = a * d - b * c
                dets.append(detJ)

                # conditioning proxy: ||J||_F / |detJ|
                fro = np.sqrt(a * a + b * b + c * c + d * d)
                kappas.append(fro / (abs(detJ) + 1e-12))

        dets = np.asarray(dets)
        kappas = np.asarray(kappas)

        # min_detJ, min_abs_detJ, frac_neg_detJ, max_kappa_proxy
        return float(dets.min()), float(np.min(np.abs(dets))), float(np.mean(dets <= 0.0)),float(kappas.max()),

    @staticmethod
    @njit(cache=True, fastmath=True)
    def distort_norm_numba(nx: float, ny: float,
                           k1: float, k2: float, k3: float,
                           p1: float, p2: float,
                           has_tangential: bool):
        """
        Forward distortion in normalized coordinates.
        Returns (nxd, nyd).
        """
        x2 = nx * nx
        y2 = ny * ny
        r2 = x2 + y2
        xy = nx * ny

        # radial factor (Horner)
        L = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))

        if has_tangential:
            dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
            dy = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy
        else:
            dx = 0.0
            dy = 0.0

        return nx * L + dx, ny * L + dy

    def distort_point(self, pixel):
        """
        Numba-accelerated version of distort_point.

        Same contract:
          - no-op if pixel already distorted
          - requires validCal
          - uses pixel.norm_coords (computes from pix if needed)
          - updates pixel.norm_coords, then updates pixel.pix_coords, flips is_undistorted False
        """
        if not pixel.is_undistorted:
            return
        if not self.validCal:
            raise ValueError("Trying to unproject points, but calibration isn't complete.")
        if pixel.pix_coords is None and pixel.norm_coords is None:
            raise ValueError("Pixel doesn't have any coordinates.")

        if pixel.norm_coords is None:
            self.havePix_needNorm(pixel)

        nx, ny = float(pixel.norm_coords[0]), float(pixel.norm_coords[1])

        nxd, nyd = self.distort_norm_numba(
            nx, ny,
            float(self.k1), float(self.k2), float(self.k3),
            float(self.p1), float(self.p2),
            bool(getattr(self, "has_tangential", True))  # or however you store it
        )

        # keep type consistent with your code (list)
        pixel.norm_coords = [float(nxd), float(nyd)]
        self.haveNorm_needPix(pixel)
        pixel.is_undistorted = False

    def undistort_point(self, pixel: pxl):
        '''
        >>> cal = default_864_cam()
        >>> p = pxl(pix_coords=(700.0, 250.0))
        >>> cal.undistort_point(p)
        >>> first = p.pix_coords.copy()
        >>> cal.undistort_point(p)
        >>> assert p.pix_coords == first
        >>> assert p.is_undistorted

        >>> cal = default_864_cam()
        >>> p = pxl(pix_coords=(700.0, 250.0))
        >>> orig = p.pix_coords.copy()
        >>> cal.undistort_point(p)
        >>> cal.distort_point(p)
        >>> err = ((p.pix_coords[0]-orig[0])**2 + (p.pix_coords[1]-orig[1])**2) ** 0.5
        >>> assert err < 1e-6

        >>> cal = default_864_cam()
        >>> p = pxl(norm_coords=(0.2, 0.1), already_undistorted=True)
        >>> orig = p.norm_coords.copy()
        >>> cal.distort_point(p)
        >>> assert (abs(p.norm_coords[0]-orig[0]) + abs(p.norm_coords[1]-orig[1])) > 0.0

        >>> cal = default_864_cam()
        >>> # start with an undistorted cam point, distort it to get a synthetic measurement
        >>> pu = pxl(norm_coords=(0.25, -0.15), already_undistorted=True)
        >>> cal.distort_point(pu)                 # now pu.norm_coords is distorted
        >>> xd, yd = pu.norm_coords
        >>> # now undistort back
        >>> cal.undistort_point(pu)
        >>> xu, yu = pu.norm_coords
        >>> # forward-distort xu,yu and compare to xd,yd
        >>> test = pxl(norm_coords=(xu, yu), already_undistorted=True)
        >>> cal.distort_point(test)
        >>> assert abs(test.norm_coords[0]-xd) < 1e-10 and abs(test.norm_coords[1]-yd) < 1e-10
        '''
        if pixel.is_undistorted:
            return
        if pixel.pix_coords is None and pixel.norm_coords is None:
            raise ValueError("Pixel doesn't have any coordinates.")
        if not self.validCal:
            raise ValueError("Trying to unproject points, but calibration isn't complete.")

        if pixel.norm_coords is None:
            self.havePix_needNorm(pixel)

        def cleanup(pixel: pxl, Nx: float, Ny: float):
            pixel.norm_coords = [Nx, Ny]
            self.haveNorm_needPix(pixel)
            pixel.is_undistorted = True

        kMinAbsL = 1e-12
        kMinRes = 1e-14
        kMinAbsDet = 1e-18

        x_d = pixel.norm_coords[0]
        y_d = pixel.norm_coords[1]

        new_x = copy.deepcopy(x_d)
        new_y = copy.deepcopy(y_d)

        # 2 Fixed Point Iterations
        def compute_L_and_tangential(_x: float, _y: float) -> tuple[float, float, float, float]:
            x2 = _x * _x
            y2 = _y * _y
            r2 = x2 + y2

            L = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))

            dL_dr2 = self.k1 + (2.0 * self.k2 + 3.0 * self.k3 * r2) * r2

            if not self.has_tangential:
                return L, dL_dr2, 0.0, 0.0

            xy = _x * _y
            dx = 2.0 * self.p1 * xy + self.p2 * (r2 + 2.0 * x2)
            dy = self.p1 * (r2 + 2.0 * y2) + 2.0 * self.p2 * xy

            return L, dL_dr2, dx, dy

        for _ in range(2):
            L, dL_dr2, dx, dy = compute_L_and_tangential(new_x, new_y)
            if abs(L) < kMinAbsL:
                cleanup(pixel, new_x, new_y)
                return

            new_x = (x_d - dx) / L
            new_y = (y_d - dy) / L

        # 1 Newton Cleanup
        L, dL_dr2, dx, dy = compute_L_and_tangential(new_x, new_y)

        x2 = new_x * new_x
        y2 = new_y * new_y

        dL_dx = 2.0 * new_x * dL_dr2
        dL_dy = 2.0 * new_y * dL_dr2

        ddx_dx = ddx_dy = ddy_dx = ddy_dy = 0.0
        if self.has_tangential:
            ddx_dx = 2.0 * self.p1 * new_y + 6.0 * self.p2 * new_x
            ddx_dy = 2.0 * self.p1 * new_x + 2.0 * self.p2 * new_y
            ddy_dx = 2.0 * self.p1 * new_x + 2.0 * self.p2 * new_y
            ddy_dy = 6.0 * self.p1 * new_y + 2.0 * self.p2 * new_x

        gx = (new_x * L + dx) - x_d
        gy = (new_y * L + dy) - y_d

        if abs(gx) + abs(gy) < kMinRes:
            cleanup(pixel, new_x, new_y)
            return

        J11 = L + new_x * dL_dx + ddx_dx
        J12 = new_x * dL_dy + ddx_dy
        J21 = new_y * dL_dx + ddy_dx
        J22 = L + new_y * dL_dy + ddy_dy

        det = J11 * J22 - J12 * J21
        if abs(det) > kMinAbsDet:
            inv_det = 1.0 / det
            del_x = (gx * J22 - gy * J12) * inv_det
            del_y = (-gx * J21 + gy * J11) * inv_det

            new_x -= del_x
            new_y -= del_y

        cleanup(pixel, new_x, new_y)

@njit(parallel=True, cache=True, fastmath=True)
def distort_points_px_numba(pts_px_und,
                            fx, fy, cx, cy,
                            k1, k2, p1, p2, k3,
                            has_tangential):
    n = pts_px_und.shape[0]
    out = np.empty((n, 2), dtype=np.float64)

    for i in prange(n):
        u = pts_px_und[i, 0]
        v = pts_px_und[i, 1]

        # pixels -> normalized
        x = (u - cx) / fx
        y = (v - cy) / fy

        x2 = x * x
        y2 = y * y
        r2 = x2 + y2
        xy = x * y

        # radial (Horner)
        L = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))

        if has_tangential:
            dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
            dy = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy
        else:
            dx = 0.0
            dy = 0.0

        xd = x * L + dx
        yd = y * L + dy

        out[i, 0] = xd * fx + cx
        out[i, 1] = yd * fy + cy

    return out

def distort_points_px(cal, pts_px_und):
    """
    Fisheye doctest: forward then inverse round-trip.

    >>> cal = default_fisheye_cam()
    >>> assert cal.validCal
    >>> pts = np.array([[100.0, 100.0],
    ...                 [432.0, 432.0],
    ...                 [800.0, 200.0]], dtype=np.float64)
    >>> d = distort_points_px(cal, pts)
    >>> u = undistort_points_px(cal, d)
    >>> float(np.max(np.sqrt(np.sum((u - pts)**2, axis=1)))) < 1e-9
    True

    Scalar input returns a 2-vector:

    >>> p = np.array([123.0, 456.0], dtype=np.float64)
    >>> q = distort_points_px(cal, p)
    >>> q.shape
    (2,)
    """
    pts = np.asarray(pts_px_und, dtype=np.float64)
    scalar = (pts.ndim == 1)
    if scalar:
        pts = pts.reshape(1, 2)

    if bool(getattr(cal, "fisheye", False)):
        out = distort_points_px_fisheye_numba(
            pts,
            float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
            float(cal.k1), float(cal.k2), float(cal.k3), float(cal.k4),
        )
    else:
        out = distort_points_px_numba(
            pts,
            float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
            float(cal.k1), float(cal.k2), float(cal.p1), float(cal.p2), float(cal.k3),
            bool(cal.has_tangential)
        )

    return out[0] if scalar else out

def _radial_terms(x, y, k1, k2, k3):
    """Compute r2, r4, L and dL/dr2, plus dL/dx,dL/dy."""
    x2 = x * x
    y2 = y * y
    r2 = x2 + y2
    r4 = r2 * r2

    # L = 1 + k1*r2 + k2*r4 + k3*r6
    # r6 = r4*r2
    L = 1.0 + k1 * r2 + k2 * r4 + k3 * (r4 * r2)

    # dL/dr2 = k1 + 2*k2*r2 + 3*k3*r2^2 = k1 + 2*k2*r2 + 3*k3*r4
    dL_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4

    dL_dx = 2.0 * x * dL_dr2
    dL_dy = 2.0 * y * dL_dr2
    return x2, y2, r2, r4, L, dL_dx, dL_dy


def _newton_update_in_place(
    x, y,
    x_d, y_d,
    L, dL_dx, dL_dy,
    gx, gy,
    do_mask,
    kMinAbsDet,
    # tangential jacobian add-ons (can be None for radial-only)
    ddx_dx=None, ddx_dy=None, ddy_dx=None, ddy_dy=None
):
    """
    Apply one Newton step in-place on x,y for the subset do_mask.
    Everything is vectorized; expects arrays of the same shape.
    """
    if not np.any(do_mask):
        return

    # Build Jacobian (with optional tangential derivative add-ons)
    if ddx_dx is None:
        J11 = L + x * dL_dx
        J12 = x * dL_dy
        J21 = y * dL_dx
        J22 = L + y * dL_dy
    else:
        J11 = L + x * dL_dx + ddx_dx
        J12 = x * dL_dy + ddx_dy
        J21 = y * dL_dx + ddy_dx
        J22 = L + y * dL_dy + ddy_dy

    det = J11 * J22 - J12 * J21
    do = do_mask & (np.abs(det) >= kMinAbsDet)
    if not np.any(do):
        return

    inv_det = 1.0 / det[do]
    del_x = (gx[do] * J22[do] - gy[do] * J12[do]) * inv_det
    del_y = (-gx[do] * J21[do] + gy[do] * J11[do]) * inv_det
    x[do] -= del_x
    y[do] -= del_y

def undistort_points_px(cal, pts_px_dist, mode="precise", eps_px=1e-14):
    """
    Fisheye doctest: inverse then forward round-trip.

    >>> cal = default_fisheye_cam()
    >>> assert cal.validCal
    >>> pts = np.array([[120.0, 140.0],
    ...                 [432.0, 432.0],
    ...                 [700.0, 820.0]], dtype=np.float64)
    >>> u = undistort_points_px(cal, pts)
    >>> d = distort_points_px(cal, u)
    >>> float(np.max(np.sqrt(np.sum((d - pts)**2, axis=1)))) < 1e-9
    True

    Optional: compare to OpenCV if installed.

    >>> try:
    ...     import cv2
    ... except Exception:
    ...     cv2 = None
    >>> if cv2 is not None:
    ...     K = np.array([[cal.fx, 0.0, cal.cx],
    ...                   [0.0, cal.fy, cal.cy],
    ...                   [0.0, 0.0, 1.0]], dtype=np.float64)
    ...     D = np.array([cal.k1, cal.k2, cal.k3, cal.k4], dtype=np.float64)
    ...     pts_cv = pts.reshape(-1, 1, 2)
    ...     ocv = cv2.fisheye.undistortPoints(pts_cv, K, D, R=None, P=K).reshape(-1, 2)
    ...     ours = undistort_points_px(cal, pts)
    ...     float(np.max(np.sqrt(np.sum((ours - ocv)**2, axis=1)))) < 1e-9
    True
    """
    pts = np.asarray(pts_px_dist, dtype=np.float64)
    scalar = (pts.ndim == 1)
    if scalar:
        pts = pts.reshape(1, 2)

    if bool(getattr(cal, "fisheye", False)):
        # For fisheye we expose multiple internal theta inversion strategies.
        # - "newton"   : pure Newton iterations (existing behavior)
        # - "precise"  : fixed-point warm start + Newton refinement
        # (You can add more strings later without touching the callers.)
        mode_l = str(mode).lower()
        use_precise = (mode_l in ("precise", "fisheye_precise", "fp_newton", "hybrid"))
        out = undistort_points_px_fisheye_numba(
            pts,
            float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
            float(cal.k1), float(cal.k2), float(cal.k3), float(cal.k4),
            bool(use_precise),
        )
    else:
        out = undistort_points_px_numba_dispatch(
            pts,
            float(cal.fx), float(cal.fy), float(cal.cx), float(cal.cy),
            float(cal.k1), float(cal.k2), float(cal.p1), float(cal.p2), float(cal.k3),
            bool(cal.has_tangential),
            mode_opencv_5fp=(mode == "opencv"),
            eps_px=float(eps_px),
        )

    return out[0] if scalar else out

@njit(parallel=True, fastmath=True)
def undistort_points_px_numba(
    pts_px_dist,
    fx, fy, cx, cy,
    k1, k2, p1, p2, k3,
    has_tangential,
    mode_opencv_5fp,         # True => 5 FP, False => 2 FP + gated Newton
    eps_px
):
    N = pts_px_dist.shape[0]
    out = np.empty((N, 2), dtype=np.float64)

    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    kMinAbsL = 1e-12
    kMinAbsDet = 1e-18

    iters = 5 if mode_opencv_5fp else 2

    for i in prange(N):
        u = pts_px_dist[i, 0]
        v = pts_px_dist[i, 1]

        x_d = (u - cx) * inv_fx
        y_d = (v - cy) * inv_fy

        x = x_d
        y = y_d

        # ----- fixed point -----
        for _ in range(iters):
            x2 = x*x
            y2 = y*y
            r2 = x2 + y2
            r4 = r2*r2
            r6 = r4*r2
            L = 1.0 + k1*r2 + k2*r4 + k3*r6
            if abs(L) < kMinAbsL:
                break

            dx = 0.0
            dy = 0.0
            if has_tangential:
                xy = x*y
                dx = 2.0*p1*xy + p2*(r2 + 2.0*x2)
                dy = p1*(r2 + 2.0*y2) + 2.0*p2*xy

            x = (x_d - dx) / L
            y = (y_d - dy) / L

        # ----- optional Newton cleanup -----
        if not mode_opencv_5fp:
            x2 = x*x
            y2 = y*y
            r2 = x2 + y2
            r4 = r2*r2
            r6 = r4*r2
            L = 1.0 + k1*r2 + k2*r4 + k3*r6
            if abs(L) >= kMinAbsL:
                dx = 0.0
                dy = 0.0
                ddx_dx = ddx_dy = ddy_dx = ddy_dy = 0.0

                if has_tangential:
                    xy = x*y
                    dx = 2.0*p1*xy + p2*(r2 + 2.0*x2)
                    dy = p1*(r2 + 2.0*y2) + 2.0*p2*xy

                    ddx_dx = 2.0*p1*y + 6.0*p2*x
                    ddx_dy = 2.0*p1*x + 2.0*p2*y
                    ddy_dx = 2.0*p1*x + 2.0*p2*y
                    ddy_dy = 6.0*p1*y + 2.0*p2*x

                gx = (x*L + dx) - x_d
                gy = (y*L + dy) - y_d

                # pixel-based gate
                res_px = (abs(gx) + abs(gy)) * (fx if fx > fy else fy)
                if res_px >= eps_px:
                    dL_dr2 = k1 + 2.0*k2*r2 + 3.0*k3*r4
                    dL_dx = 2.0*x*dL_dr2
                    dL_dy = 2.0*y*dL_dr2

                    if has_tangential:
                        J11 = L + x*dL_dx + ddx_dx
                        J12 = x*dL_dy + ddx_dy
                        J21 = y*dL_dx + ddy_dx
                        J22 = L + y*dL_dy + ddy_dy
                    else:
                        J11 = L + x*dL_dx
                        J12 = x*dL_dy
                        J21 = y*dL_dx
                        J22 = L + y*dL_dy

                    det = J11*J22 - J12*J21
                    if abs(det) >= kMinAbsDet:
                        inv_det = 1.0 / det
                        del_x = (gx*J22 - gy*J12) * inv_det
                        del_y = (-gx*J21 + gy*J11) * inv_det
                        x -= del_x
                        y -= del_y

        out[i, 0] = x*fx + cx
        out[i, 1] = y*fy + cy

    return out


# -------------------------- constants (compile-time) --------------------------
_KMIN_ABS_L = 1e-12
_KMIN_ABS_DET = 1e-18

@njit(parallel=True, fastmath=True, cache=True)
def undistort_5fp_no_tan(pts_px_dist, fx, fy, cx, cy, k1, k2, k3):
    N = pts_px_dist.shape[0]
    out = np.empty((N, 2), dtype=np.float64)
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    for i in prange(N):
        u = pts_px_dist[i, 0]
        v = pts_px_dist[i, 1]

        x_d = (u - cx) * inv_fx
        y_d = (v - cy) * inv_fy

        x = x_d
        y = y_d

        # 5 fixed-point iterations
        for _ in range(5):
            x2 = x * x
            y2 = y * y
            r2 = x2 + y2
            r4 = r2 * r2
            r6 = r4 * r2
            L = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            if abs(L) < _KMIN_ABS_L:
                break
            x = x_d / L
            y = y_d / L

        out[i, 0] = x * fx + cx
        out[i, 1] = y * fy + cy

    return out


@njit(parallel=True, fastmath=True, cache=True)
def undistort_5fp_tan(pts_px_dist, fx, fy, cx, cy, k1, k2, p1, p2, k3):
    N = pts_px_dist.shape[0]
    out = np.empty((N, 2), dtype=np.float64)
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    for i in prange(N):
        u = pts_px_dist[i, 0]
        v = pts_px_dist[i, 1]

        x_d = (u - cx) * inv_fx
        y_d = (v - cy) * inv_fy

        x = x_d
        y = y_d

        # 5 fixed-point iterations
        for _ in range(5):
            x2 = x * x
            y2 = y * y
            r2 = x2 + y2
            r4 = r2 * r2
            r6 = r4 * r2
            L = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            if abs(L) < _KMIN_ABS_L:
                break

            xy = x * y
            dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
            dy = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy

            x = (x_d - dx) / L
            y = (y_d - dy) / L

        out[i, 0] = x * fx + cx
        out[i, 1] = y * fy + cy

    return out


# -------------------------- 2FP + gated Newton -------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def undistort_2fp_newton_no_tan(pts_px_dist, fx, fy, cx, cy, k1, k2, k3, eps_px):
    N = pts_px_dist.shape[0]
    out = np.empty((N, 2), dtype=np.float64)
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy
    scale = fx if fx > fy else fy

    for i in prange(N):
        u = pts_px_dist[i, 0]
        v = pts_px_dist[i, 1]

        x_d = (u - cx) * inv_fx
        y_d = (v - cy) * inv_fy

        x = x_d
        y = y_d

        # 2 fixed-point iterations
        for _ in range(2):
            x2 = x * x
            y2 = y * y
            r2 = x2 + y2
            r4 = r2 * r2
            r6 = r4 * r2
            L = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            if abs(L) < _KMIN_ABS_L:
                break
            x = x_d / L
            y = y_d / L

        # one Newton step, gated
        x2 = x * x
        y2 = y * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2
        L = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        if abs(L) >= _KMIN_ABS_L:
            gx = (x * L) - x_d
            gy = (y * L) - y_d

            res_px = (abs(gx) + abs(gy)) * scale
            if res_px >= eps_px:
                dL_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
                dL_dx = 2.0 * x * dL_dr2
                dL_dy = 2.0 * y * dL_dr2

                # Jacobian of g(x,y) = [xL - xd, yL - yd]
                J11 = L + x * dL_dx
                J12 = x * dL_dy
                J21 = y * dL_dx
                J22 = L + y * dL_dy

                det = J11 * J22 - J12 * J21
                if abs(det) >= _KMIN_ABS_DET:
                    inv_det = 1.0 / det
                    del_x = (gx * J22 - gy * J12) * inv_det
                    del_y = (-gx * J21 + gy * J11) * inv_det
                    x -= del_x
                    y -= del_y

        out[i, 0] = x * fx + cx
        out[i, 1] = y * fy + cy

    return out


@njit(parallel=True, fastmath=True, cache=True)
def undistort_2fp_newton_tan(pts_px_dist, fx, fy, cx, cy, k1, k2, p1, p2, k3, eps_px):
    N = pts_px_dist.shape[0]
    out = np.empty((N, 2), dtype=np.float64)
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy
    scale = fx if fx > fy else fy

    for i in prange(N):
        u = pts_px_dist[i, 0]
        v = pts_px_dist[i, 1]

        x_d = (u - cx) * inv_fx
        y_d = (v - cy) * inv_fy

        x = x_d
        y = y_d

        # 2 fixed-point iterations
        for _ in range(2):
            x2 = x * x
            y2 = y * y
            r2 = x2 + y2
            r4 = r2 * r2
            r6 = r4 * r2
            L = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            if abs(L) < _KMIN_ABS_L:
                break

            xy = x * y
            dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
            dy = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy

            x = (x_d - dx) / L
            y = (y_d - dy) / L

        # one Newton step, gated
        x2 = x * x
        y2 = y * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2
        L = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        if abs(L) >= _KMIN_ABS_L:
            xy = x * y
            dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
            dy = p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy

            gx = (x * L + dx) - x_d
            gy = (y * L + dy) - y_d

            res_px = (abs(gx) + abs(gy)) * scale
            if res_px >= eps_px:
                dL_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
                dL_dx = 2.0 * x * dL_dr2
                dL_dy = 2.0 * y * dL_dr2

                # tangential partials
                ddx_dx = 2.0 * p1 * y + 6.0 * p2 * x
                noted = 2.0 * p1 * x + 2.0 * p2 * y
                ddx_dy = noted
                ddy_dx = noted
                ddy_dy = 6.0 * p1 * y + 2.0 * p2 * x

                J11 = L + x * dL_dx + ddx_dx
                J12 = x * dL_dy + ddx_dy
                J21 = y * dL_dx + ddy_dx
                J22 = L + y * dL_dy + ddy_dy

                det = J11 * J22 - J12 * J21
                if abs(det) >= _KMIN_ABS_DET:
                    inv_det = 1.0 / det
                    del_x = (gx * J22 - gy * J12) * inv_det
                    del_y = (-gx * J21 + gy * J11) * inv_det
                    x -= del_x
                    y -= del_y

        out[i, 0] = x * fx + cx
        out[i, 1] = y * fy + cy

    return out

@njit(cache=True, fastmath=True)
def _fisheye_theta_from_theta_d(theta_d, k1, k2, k3, k4):
    """
    Solve theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8) = theta_d
    via Newton iterations.
    """
    if theta_d <= 1e-24:
        return 0.0

    # Good initial guess: theta ~= theta_d
    theta = theta_d

    # Newton iterations (OpenCV does something similar internally)
    for _ in range(8):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4

        poly = 1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8
        f = theta * poly - theta_d

        # derivative:
        # d/dtheta [theta*poly] = poly + theta * dpoly/dtheta
        # dpoly/dtheta = 2*k1*theta + 4*k2*theta^3 + 6*k3*theta^5 + 8*k4*theta^7
        dpoly = (2.0 * k1 * theta
                 + 4.0 * k2 * theta * t2
                 + 6.0 * k3 * theta * t4
                 + 8.0 * k4 * theta * t6)
        fp = poly + theta * dpoly

        if abs(fp) < 1e-24:
            break

        step = f / fp
        theta -= step

        if abs(step) < 1e-14:
            break

    return theta

@njit(cache=True, fastmath=True)
def _fisheye_theta_from_theta_d_precise(theta_d, k1, k2, k3, k4):
    """
    Solve theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8) = theta_d

    Uses:
      - Fixed-point iterations for a safe warm start
      - Newton iterations for fast convergence
    """
    if theta_d <= 1e-24:
        return 0.0

    # ------------------------------------------------------------
    # Fixed-point warm start
    # ------------------------------------------------------------
    theta = theta_d  # good initial guess for small angles

    for _ in range(2):  # hard-coded FP iters (tune later)
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4

        denom = 1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8

        # Prevent division blow-up
        if abs(denom) < 1e-24:
            break

        theta = theta_d / denom

    # ------------------------------------------------------------
    # Newton refinement (as you already had)
    # ------------------------------------------------------------
    for _ in range(8):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4

        poly = 1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8
        f = theta * poly - theta_d

        dpoly = (2.0 * k1 * theta
                 + 4.0 * k2 * theta * t2
                 + 6.0 * k3 * theta * t4
                 + 8.0 * k4 * theta * t6)
        fp = poly + theta * dpoly

        if abs(fp) < 1e-24:
            break

        step = f / fp
        theta -= step

        if abs(step) < 1e-14:
            break

    return theta


@njit(parallel=True, cache=True, fastmath=True)
def undistort_points_px_fisheye_numba(
    pts_px_dist,
    fx, fy, cx, cy,
    k1, k2, k3, k4,
    use_precise: bool,
):
    """
    Inverse of OpenCV fisheye distortion model.
    Input: distorted pixel points (N,2)
    Output: undistorted pixel points (N,2)
    """
    N = pts_px_dist.shape[0]
    out = np.empty((N, 2), dtype=np.float64)

    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    for i in prange(N):
        u = pts_px_dist[i, 0]
        v = pts_px_dist[i, 1]

        # pixels -> distorted normalized
        xd = (u - cx) * inv_fx
        yd = (v - cy) * inv_fy

        rd = np.sqrt(xd * xd + yd * yd)

        if rd <= 1e-24:
            # center stays center
            x = xd
            y = yd
        else:
            theta_d = rd
            if use_precise:
                theta = _fisheye_theta_from_theta_d_precise(theta_d, k1, k2, k3, k4)
            else:
                theta = _fisheye_theta_from_theta_d(theta_d, k1, k2, k3, k4)
            r = np.tan(theta)

            scale = r / rd
            x = xd * scale
            y = yd * scale

        out[i, 0] = x * fx + cx
        out[i, 1] = y * fy + cy

    return out


@njit(parallel=True, cache=True, fastmath=True)
def distort_points_px_fisheye_numba(pts_px_und, fx, fy, cx, cy, k1, k2, k3, k4):
    """
    Forward OpenCV fisheye distortion model.
    Input: undistorted pixel points (N,2)
    Output: distorted pixel points (N,2)
    """
    N = pts_px_und.shape[0]
    out = np.empty((N, 2), dtype=np.float64)

    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    for i in prange(N):
        u = pts_px_und[i, 0]
        v = pts_px_und[i, 1]

        # pixels -> undistorted normalized
        x = (u - cx) * inv_fx
        y = (v - cy) * inv_fy

        r = np.sqrt(x * x + y * y)
        if r <= 1e-24:
            xd = x
            yd = y
        else:
            theta = np.arctan(r)
            t2 = theta * theta
            t4 = t2 * t2
            t6 = t4 * t2
            t8 = t4 * t4

            theta_d = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)
            scale = theta_d / r

            xd = x * scale
            yd = y * scale

        out[i, 0] = xd * fx + cx
        out[i, 1] = yd * fy + cy

    return out
# -------------------------- dispatcher (Python) -------------------------------

def undistort_points_px_numba_dispatch(
    pts_px_dist,
    fx, fy, cx, cy,
    k1, k2, p1, p2, k3,
    has_tangential: bool,
    mode_opencv_5fp: bool,
    eps_px: float
):
    pts = np.asarray(pts_px_dist, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts_px_dist must be (N,2)")

    if mode_opencv_5fp:
        if has_tangential:
            return undistort_5fp_tan(pts, fx, fy, cx, cy, k1, k2, p1, p2, k3)
        else:
            return undistort_5fp_no_tan(pts, fx, fy, cx, cy, k1, k2, k3)
    else:
        if has_tangential:
            return undistort_2fp_newton_tan(pts, fx, fy, cx, cy, k1, k2, p1, p2, k3, eps_px)
        else:
            return undistort_2fp_newton_no_tan(pts, fx, fy, cx, cy, k1, k2, k3, eps_px)

def default_864_cam():
    cal = Calibration()
    cal.fx = cal.fy = 941.75
    cal.cx = cal.cy = 432.0
    cal.k1 = -0.186
    cal.k2 = 0.137
    cal.p1 = -0.000232
    cal.p2 = 0.000432
    cal.k3 = -0.0137
    cal.calTime = 0.0
    cal.numCBUsed = 0
    cal.rmsError = 0.0
    cal.hfov = 0.0
    cal.width = 864
    cal.height = 864
    return cal

def default_fisheye_cam():
    cal = Calibration()
    cal.fisheye = True
    cal.fx = cal.fy = 450.0
    cal.cx = cal.cy = 432.0
    cal.k1 = -0.010
    cal.k2 = 0.0015
    cal.k3 = -0.0002
    cal.k4 = 0.00002
    cal.calTime = 0.0
    cal.numCBUsed = 0
    cal.rmsError = 0.0
    cal.hfov = 0.0
    cal.width = 864
    cal.height = 864
    return cal

def default_2848_cam():
    cal = Calibration()
    cal.fx = cal.fy = 3085.026
    cal.cx = cal.cy = 1423.5
    cal.k1 = -0.187
    cal.k2 = 0.137
    cal.p1 = -0.000232
    cal.p2 = 0.000432
    cal.k3 = -0.000269
    cal.calTime = 0.0
    cal.numCBUsed = 0
    cal.rmsError = 0.0
    cal.hfov = 0.0
    cal.width = 2848
    cal.height = 2848
    return cal


if __name__ == "__main__":
    import time
    import numpy as np

    try:
        import cv2
        _HAS_CV2 = True
        cv2.setUseOptimized(True)
    except Exception:
        _HAS_CV2 = False

    def _K_D_from_cal(cal):
        K = np.array([[cal.fx, 0.0, cal.cx],
                      [0.0, cal.fy, cal.cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        if bool(getattr(cal, "fisheye", False)):
            D = np.array([cal.k1, cal.k2, cal.k3, cal.k4], dtype=np.float64)
        else:
            # OpenCV undistortPoints expects up to 8; you probably only use k1,k2,p1,p2,k3
            # If you don’t have tangential, p1/p2 should be 0 already.
            k1 = float(getattr(cal, "k1", 0.0))
            k2 = float(getattr(cal, "k2", 0.0))
            p1 = float(getattr(cal, "p1", 0.0))
            p2 = float(getattr(cal, "p2", 0.0))
            k3 = float(getattr(cal, "k3", 0.0))
            D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        return K, D

    def _make_points(cal, N=200_000, seed=0):
        rng = np.random.default_rng(seed)
        pts = np.empty((N, 2), dtype=np.float64)
        pts[:, 0] = rng.random(N) * (float(cal.width) - 1.0)
        pts[:, 1] = rng.random(N) * (float(cal.height) - 1.0)
        return pts

    def _bench(label, fn, pts, warmup=2, reps=10):
        # Warmup (important for numba + caches)
        for _ in range(warmup):
            _ = fn(pts)

        times = np.empty(reps, dtype=np.float64)
        for i in range(reps):
            t0 = time.perf_counter()
            _ = fn(pts)
            t1 = time.perf_counter()
            times[i] = t1 - t0

        med = float(np.median(times))
        p10 = float(np.percentile(times, 10))
        p90 = float(np.percentile(times, 90))
        N = pts.shape[0]
        print(f"{label:28s}  med={med*1000:7.3f} ms  p10={p10*1000:7.3f} ms  p90={p90*1000:7.3f} ms  ({N/med:,.1f} pts/s)")

    def benchmark_undistort(cal, N=200_000, seed=0):
        print("\n==============================")
        print(f"Benchmark undistortPoints | fisheye={bool(getattr(cal,'fisheye',False))} | N={N}")
        print("==============================")

        pts = _make_points(cal, N=N, seed=seed)

        # Your Numba-routed implementation
        def ours(pts_in):
            return undistort_points_px(cal, pts_in)

        _bench("OURS (numba) undistort", ours, pts)

        if not _HAS_CV2:
            print("OpenCV not available; skipping cv2 benchmark.")
            return

        K, D = _K_D_from_cal(cal)

        # OpenCV expects Nx1x2
        pts_cv = pts.reshape(-1, 1, 2).astype(np.float64, copy=False)

        if bool(getattr(cal, "fisheye", False)):
            def ocv(pts_in):
                pts_in_cv = pts_in.reshape(-1, 1, 2)
                out = cv2.fisheye.undistortPoints(pts_in_cv, K, D, R=None, P=K)
                return out.reshape(-1, 2)
            _bench("CV fisheye.undistortPoints  ", ocv, pts)

        else:
            def ocv(pts_in):
                pts_in_cv = pts_in.reshape(-1, 1, 2)
                out = cv2.undistortPoints(pts_in_cv, K, D, R=None, P=K)
                return out.reshape(-1, 2)
            _bench("OpenCV undistortPoints", ocv, pts)

    # --- Run benchmarks on both your standard + fisheye cal ---
    cal_std = default_864_cam()

    cal_fish = default_fisheye_cam()

    benchmark_undistort(cal_std,  N=200_000, seed=1)
    benchmark_undistort(cal_fish, N=200_000, seed=2)

