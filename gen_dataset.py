import numpy as np

def get_pose_2d_dataset_1(
    N1=20, N2=25, N3=20,
    step_size=1.0,
    odom_sigma=(0.1, 0.1, 0.05),   # x,y,theta noise for odometry
    gps_sigma=(1, 1),          # x,y noise for gps
    gps_rate=5,                    # one gps every gps_rate poses
    seed=42
):
    """
    Generate synthetic Pose2 dataset:
    - Ground truth trajectory: straight N1 steps, turn left then N2 steps, then right then N3 steps.
    """
    rng = np.random.default_rng(seed)

    # 1. Ground truth trajectory
    gt_poses = []
    x, y, theta = 0.0, 0.0, 0.0

    for _ in range(N1):
        gt_poses.append((x, y, theta))
        x += step_size
    # turn left
    theta = np.pi/2
    for _ in range(N2):
        gt_poses.append((x, y, theta))
        y += step_size

    theta = 0.0
    for _ in range(N3):
        gt_poses.append((x, y, theta))
        x += step_size

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

    np.savez("data/synthetic_pose_2d_dataset_1.npz",
             gt_poses=gt_poses,
             odom_meas=odom_meas,
             gps_meas=gps_meas)

    return gt_poses, odom_meas, gps_meas

if __name__ == "__main__":
    gt, odom, gps = get_pose_2d_dataset_1()
    print("Ground truth poses:", gt.shape)
    print("Odometry measurements:", odom.shape)
    print("GPS measurements:", gps.shape)
