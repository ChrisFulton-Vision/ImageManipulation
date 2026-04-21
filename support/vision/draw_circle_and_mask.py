import numpy as np
import cv2

def dim_except_circle(frame, center, x_axes, y_axes=None, dim_factor=0.5) -> None:
    """
    Dims an image everywhere except inside a circle.

    Args:
        frame (np.array): the image
        center (tuple): (x, y) coordinates of the circle's center.
        dim_factor (float): Dimming factor (0 to 1, 0 for black, 1 for no dimming).
    """

    if y_axes is None:
        radius = x_axes
        if dim_factor == 0.0:
            dim_entirely(frame, center, radius)
            return

        # 1. Create a mask
        mask = np.zeros(frame.shape[:2], dtype="uint8")  # Black mask
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)  # White circle on mask

    else:
        mask = np.zeros(frame.shape[:2], dtype='uint8')
        # rectangle(mask, (int(center[0]-x_axes),int(center[1]-y_axes)),(int(center[0]+x_axes),int(center[1]+y_axes)),
        #               color=255, thickness=-1)
        cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(x_axes), int(y_axes)),
                    angle=0, startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-1)

    # 2. Dim the entire image
    dimmed_img = (frame * dim_factor).astype("uint8")

    # 3. Copy the original circle area back to the dimmed image
    masked_circle = cv2.bitwise_and(frame, frame, mask=mask)

    # Invert the mask to select the area outside the circle
    inverted_mask = cv2.bitwise_not(mask)

    # Apply the mask to the dimmed image
    masked_dimmed = cv2.bitwise_and(dimmed_img, dimmed_img, mask=inverted_mask)

    # Add the original circle back
    cv2.add(masked_circle, masked_dimmed, dst=frame)

    return


def dim_entirely(frame, center, radius):
    """
    Dims an image everywhere except inside a circle.

    Args:
        frame (np.array): the image
        center (tuple): (x, y) coordinates of the circle's center.
        radius (int): Radius of the circle.
    """

    # 1. Create a mask
    mask = np.zeros(frame.shape[:2], dtype="uint8")  # Black mask
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), -1)  # White circle on mask

    # 3. Copy the original circle area back to the dimmed image
    cv2.bitwise_and(frame, frame, dst=frame, mask=mask)


