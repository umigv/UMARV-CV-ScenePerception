import h5py
import ransac
import cv2
import numpy as np
import os

iters = 50
kernel = (1, 16)  # kernel is rows, columns
tolerance = 0.1

filename = "res/19_11_10.hdf5"
fourcc = cv2.VideoWriter_fourcc(*"XVID")


def main():

    f = h5py.File(filename, "r")
    color = f["images"]
    depth = f["depth_maps"]

    if len(depth) < 1:
        return
    h, w = depth[0].shape

    try:
        os.mkdir("out")
    except(FileExistsError):
        pass

    writer = cv2.VideoWriter("out/ransac.avi", fourcc, 30, (w, 2 * h))

    for i in range(len(depth) // 1):
        mask = (255 * ransac.ransac(depth[i], iters, kernel, tolerance)).astype(
            np.uint8
        )
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.rectangle(mask, (50, 100), (w-50, h-2), (0, 255, 0), 2)
        mask = cv2.putText(mask, "reliable area", (75, 125), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        view = color[i][:, : int(color[i].shape[1] / 2)]
        vis = np.concatenate((view, mask), axis=0)

        writer.write(vis)

    cv2.destroyAllWindows()
    writer.release()

    os.system("ffmpeg -i out/ransac.avi out/ransac.mp4")
    os.remove("out/ransac.avi")


if __name__ == "__main__":
    main()
