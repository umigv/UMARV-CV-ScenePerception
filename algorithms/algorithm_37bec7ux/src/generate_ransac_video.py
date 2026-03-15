import h5py
import ransac.plane
import cv2
import numpy as np
import os
import math

iters = 50
kernel = (1, 16)  # kernel is rows, columns
tolerance = 0.14

filename = "res/19_11_10.hdf5" #"res/perspective_test.svo2.hdf5"
fourcc = cv2.VideoWriter_fourcc(*"XVID")


def main():

    f = h5py.File(filename, "r")
    color = f["images"]
    depth_maps = f["depth_maps"]

    if len(depth_maps) < 1:
        return
    h, w = depth_maps[0].shape

    try:
        os.mkdir("out")
    except FileExistsError:
        pass

    writer = cv2.VideoWriter("out/ransac.avi", fourcc, 30, (w, 2 * h))

    for i in range(len(depth_maps) // 1):
        view = color[i][:, : int(color[i].shape[1] / 2)]

        depth_map = ransac.plane.clean_depths(depth_maps[i])

        masked, c = ransac.plane.hsv_and_ransac(
            view, depth_map, iters, kernel, tolerance
        )

        masked = (255 * masked).astype(np.uint8)
        masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        masked = cv2.rectangle(masked, (50, 100), (w - 50, h - 2), (0, 255, 0), 2)
        masked = cv2.putText(
            masked,
            "reliable area",
            (75, 125),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        masked = cv2.putText(
            masked,
            f"{c[0]:+.3e} {c[1]:+.3e} {c[2]:+.3e}",
            (75, 150),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        intrinsics = ransac.CameraIntrinsics(w / 2, h / 2, 600 / 2, 600 / 2)
        real_coeffs = ransac.plane.real_coeffs(c, intrinsics)
        rad = ransac.plane.real_angle(real_coeffs)
        masked = cv2.putText(
            masked,
            f"{real_coeffs[0]:+.3e} {real_coeffs[1]:+.3e} {real_coeffs[2]:+.3e}",
            (75, 175),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        masked = cv2.putText(
            masked,
            f"angle: {math.degrees(rad)}",
            (75, 200),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        masked = cv2.putText(
            masked,
            f"components: {math.cos(rad):+.3}y + {math.sin(rad):+.3}z",
            (75, 225),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        vis = np.concatenate((view, masked), axis=0)

        writer.write(vis)

    cv2.destroyAllWindows()
    writer.release()

    os.system("ffmpeg -y -i out/ransac.avi out/ransac.mp4")
    os.remove("out/ransac.avi")


if __name__ == "__main__":
    main()
