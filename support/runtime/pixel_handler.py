from collections.abc import Sequence
from typing_extensions import Optional

class Pixel:
    def __init__(self, already_undistorted: bool = False,
                    *, pix_coords: Optional[Sequence[float]] = None,
                       norm_coords: Optional[Sequence[float]]=None):

        self.is_undistorted = already_undistorted

        # These are the pixels of a specific feature or object
        self.pix_coords = None

        # These are the OpenCV style normalized coordiants of an object
        # Nx = (px - cx) / fx
        # Ny = (py - cy) / fy
        # The calibration module holds the hammer on transforms.
        # This class is therefore a COMPANION CLASS to the Calibration Module.
        self.norm_coords = None

        if pix_coords is not None and norm_coords is None:
            self.pix_coords = list(pix_coords)
            return

        if norm_coords is not None and pix_coords is None:
            self.norm_coords = list(norm_coords)
            return

        raise ValueError(f'You provided {pix_coords} and {norm_coords} but only one must not be None')

    def __str__(self):
        output = ''
        if self.norm_coords is not None:
            output += f'Nx:{self.norm_coords[0]:.3f}, Ny:{self.norm_coords[1]:.3f}, '
        if self.pix_coords is not None:
            output += f'px:{self.pix_coords[0]:.3f}, py:{self.pix_coords[1]:.3f}, '
        output += ('Undistorted' if self.is_undistorted else 'Distorted')
        return output

def main():
    pix = Pixel(norm_coords=(0.5, 0.5))
    print(pix)

if __name__ == '__main__':
    main()
