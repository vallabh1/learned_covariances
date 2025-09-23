import numpy as np
import matplotlib.pyplot as plt
def get_pose_2d_dataset_1(
    path,
    idx,
    step_size=1.0,
    odom_sigma=(0.1, 0.1, 0.05),   # x,y,theta noise for odometry
    gps_sigma=(1, 1),          # x,y noise for gps
    gps_rate=5,                    # one gps every gps_rate poses
    seed=42,
    ):
    """
    Generate synthetic Pose2 dataset:
    - Ground truth trajectory: straight N1 steps, turn left then N2 steps, then right then N3 steps.
    """
    rng = np.random.default_rng(seed)

    # 1. Ground truth trajectory
    gt_poses = []
    x, y, theta = 0.0, 0.0, 0.0

    for i in range(len(path)):
        if path[i] == "right":
            theta += np.pi/2
        elif path[i] == "left":
            theta -= np.pi/2
        else:
            for _ in range(path[i]):
                x += step_size * np.cos(theta)
                y += step_size * np.sin(theta)
                gt_poses.append((x, y, theta))
    gt_poses = np.array(gt_poses)

    odom_meas = []
    for i in range(len(gt_poses)-1):
        dx = gt_poses[i+1,0] - gt_poses[i,0]
        dy = gt_poses[i+1,1] - gt_poses[i,1]
        dtheta = gt_poses[i+1,2] - gt_poses[i,2]
        # Add 0 mean Gaussian noise
        noisy_dx = dx + rng.normal(0, odom_sigma[0])
        noisy_dy = dy + rng.normal(0, odom_sigma[1])
        noisy_dtheta = dtheta + rng.normal(0, odom_sigma[2])
        odom_meas.append((i, i+1, noisy_dx, noisy_dy, noisy_dtheta))
    odom_meas = np.array(odom_meas, dtype=float)

    gps_meas = []
    for i in range(0, len(gt_poses), gps_rate):
        noisy_x = gt_poses[i,0] + rng.normal(0, gps_sigma[0])
        noisy_y = gt_poses[i,1] + rng.normal(0, gps_sigma[1])
        gps_meas.append((i, noisy_x, noisy_y))
    gps_meas = np.array(gps_meas, dtype=float)

    np.savez(f"data/pose2d/synthetic_pose_2d_dataset_{idx}.npz",
             gt_poses=gt_poses,
             odom_meas=odom_meas,
             gps_meas=gps_meas)
    # visualize path with meters as unit
    plt.figure(figsize=(10, 10))
    plt.plot(gt_poses[:,0], gt_poses[:,1], "k-")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(f"Synthetic Pose2 Dataset {idx}")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(f"data/pose2d/synthetic_pose_2d_dataset_{idx}.png")
    plt.close()

    return gt_poses, odom_meas, gps_meas

if __name__ == "__main__":
    path1 = [40, "right", 45, "left", 40, "right", 45, "left", 40, "left", 5, "right", 2, "left", 5, "right", 2, "left", 5, "right", 2, "left", 5, "right", 2, "left", 5, "right", 2, "left", 5, "right", 2, "left"]
    path2 = [10, "left", 10, "left", 20, "left", 25, "left", 50, "left", 20, "left", 25, "left", 50, "left", 20, "left", 25, "left", 5, "left", 20, "left", 5, "left", 5, "left", 20, "left", 5, "left", 5, "left", 20]
    path3 = [20, "right", 25, "left", 50, "left", 20, "right", 80, "left", 20, "right", 80, "left", 20, "right", 80]
    path4 = [10, "left", 5, "right", 30, "right", 50, "left", 10, "left", 20, "right", 5, "right", 15, "left", 10, "right", 5, "left", 10]
    path5 = [20, "right", 25, "left", 50, "left", 20, "right", 80, "left", 20, "left", 5, "right", 5, "left", 10, "right", 5, "right", 20]
    path6 = [5, "right", 15, "left", 30, "left", 25, "right", 40, "right", 60, "left", 10, "left", 5, "right", 50, "right", 70,
             "left", 20, "left", 15, "right", 35, "left", 45, "right", 90, "left", 5, "left", 10, "right", 20, "right", 65, "left", 
             30, "right", 100]

    test_path1 =  [20, "left", 25, "right", 50, "right", 20, "left", 80, "left", 20, "left", 5, "right", 5, "left", 10, "left", 5, "right", 20]
    test_path2 =  [30, "right", 5, "left", 40, "left", 10, "left", 20, "left", 10, "left", 5, "left", 5, "right", 10, "right", 5, "left", 40]
    test_path3 = [25, "left", 10, "left", 35, "right", 5, "right", 55, "left", 75, "right", 15, "left", 5, "left", 20, "right", 45, 
                "left", 65, "right", 85, "left", 15, "right", 5, "right", 30, "left", 50, "left", 70, "right", 95, "left", 20, "right", 
                10, "left", 40, "right", 60]
    test_path4 = [15, "right", 20, "left", 10, "right", 30, "left", 5, "left", 40, "right", 50, "left", 20, "right", 15, "right", 25, 
                "left", 60, "right", 10, "left", 5]
    test_path5 = [60, "left", 10, "right", 5, "right", 15, "left", 25, "right", 30, "left", 40, "left", 5, "right", 50, "left", 20, 
                "right", 70, "left", 10, "right", 5, "left", 30, "right", 80]

    paths = [path1, path2, path3, path4, path5, path6]
    test_paths = [test_path1, test_path2, test_path3, test_path4, test_path5]
    for idx, path in enumerate(paths+test_paths):
        gt, odom, gps = get_pose_2d_dataset_1(path=path, idx=idx+1)
        print("Ground truth poses:", gt.shape)
        print("Odometry measurements:", odom.shape)
        print("GPS measurements:", gps.shape)

