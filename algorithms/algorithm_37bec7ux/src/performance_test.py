import pyzed.sl as sl
import numpy as np
import cv2
import time
import torch
import ransac_pt as ransac
import ransac_pt.plane
import ransac_pt.occu


def main():
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.async_image_retrieval = False

    devices = sl.Camera.get_device_list()
    if len(devices) < 2:
        print("Need at least 2 cameras")
        return

    cams = []
    for dev in devices[:2]:
        cam = sl.Camera()
        init.set_from_serial_number(dev.serial_number)
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open camera")
            return
        cams.append(cam)

    runtime = sl.RuntimeParameters()

    cam_info = cams[0].get_camera_information()
    resolution = cam_info.camera_configuration.resolution

    w = min(720, resolution.width)
    h = min(404, resolution.height)
    low_res = sl.Resolution(w, h)

    calib = cam_info.camera_configuration.calibration_parameters
    intr = ransac.Intrinsics(w / 2, h / 2,
                             calib.left_cam.fx / 2,
                             calib.left_cam.fy / 2)

    grid_conf = ransac.GridConfiguration(5000, 5000, 50, thres=2)

    image_mats = [sl.Mat(), sl.Mat()]
    depth_mats = [sl.Mat(), sl.Mat()]

    print("Running... press q to quit")

    prev_time = time.time()

    while True:
        occ_grids = []

        for i in range(2):

            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)

            start_depth = torch.cuda.Event(enable_timing=True)
            end_depth = torch.cuda.Event(enable_timing=True)

            start_plane = torch.cuda.Event(enable_timing=True)
            end_plane = torch.cuda.Event(enable_timing=True)

            start_occ = torch.cuda.Event(enable_timing=True)
            end_occ = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_total.record()

            if cams[i].grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            # ---------------- Depth Retrieval ----------------
            start_depth.record()
            cams[i].retrieve_measure(
                depth_mats[i], sl.MEASURE.DEPTH, sl.MEM.GPU, low_res)
            depths = ransac.plane.clean_depths(
                depth_mats[i].get_data(sl.MEM.GPU))
            end_depth.record()

            # ---------------- Plane RANSAC ----------------
            start_plane.record()
            ransac_output, px_coeffs = ransac.plane.ground_plane(
                depths, 60, (1, 16), 0.15
            )
            real_coeffs = ransac.plane.real_coeffs(px_coeffs, intr)
            end_plane.record()

            # ---------------- Occupancy ----------------
            start_occ.record()
            drive_ppc = ransac.occu.create_point_cloud(ransac_output, depths)
            drive_rpc = ransac.occu.pixel_to_real(drive_ppc, real_coeffs, intr)
            occ = ransac.occu.occupancy_grid(drive_rpc, grid_conf)
            end_occ.record()

            end_total.record()
            torch.cuda.synchronize()

            depth_time = start_depth.elapsed_time(end_depth)
            plane_time = start_plane.elapsed_time(end_plane)
            occ_time = start_occ.elapsed_time(end_occ)
            total_time = start_total.elapsed_time(end_total)

            print(f"[Cam {i}] Depth: {depth_time:.2f} ms | "
                  f"Plane: {plane_time:.2f} ms | "
                  f"Occ: {occ_time:.2f} ms | "
                  f"Total: {total_time:.2f} ms")

            occ_grids.append(occ)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        for i in range(2):
            occ_np = occ_grids[i].cpu().numpy()

            if occ_np.dtype != np.uint8:
                occ_np = occ_np.astype(np.uint8) * 255

            occ_vis = cv2.resize(
                occ_np, (600, 600),
                interpolation=cv2.INTER_NEAREST_EXACT
            )

            occ_vis = cv2.cvtColor(occ_vis, cv2.COLOR_GRAY2BGR)

            cv2.putText(
                occ_vis,
                f"Cam {i} FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow(f"Occupancy {i}", occ_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cams:
        cam.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
