import numpy as np
import gtsam

def build_and_solve_pose_2d_fg(cov_params, dataset_path="synthetic_pose_2d_dataset_1.npz"):
    """
    Build and solve a Pose2 SLAM graph with given covariance params.
    
    cov_params = (odom_sigma, gps_sigma)
      - odom_sigma: [σx, σy, σθ]
      - gps_sigma: [σx, σy]
    """
    data = np.load(f"data/pose2d/{dataset_path}", allow_pickle=True)
    gt_poses = data["gt_poses"]
    odom_meas = data["odom_meas"]
    gps_meas = data["gps_meas"]

    odom_sigma, gps_sigma = cov_params

    # Noise models
    prior_model = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
    odom_model  = gtsam.noiseModel.Diagonal.Sigmas(odom_sigma)
    gps_model   = gtsam.noiseModel.Diagonal.Sigmas(gps_sigma)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(*gt_poses[0]), prior_model))
    initial.insert(0, gtsam.Pose2(0.0, 0.0, 0.0))  

    # odometry factors
    for (i, j, dx, dy, dtheta) in odom_meas:
        graph.add(gtsam.BetweenFactorPose2(int(i), int(j),
                                           gtsam.Pose2(dx, dy, dtheta),
                                           odom_model))
        if not initial.exists(int(j)):
            prev = initial.atPose2(int(i))
            initial.insert(int(j), prev.compose(gtsam.Pose2(dx, dy, dtheta)))

    # GPS factors
    for (i, gx, gy) in gps_meas:
        graph.add(gtsam.PriorFactorPose2(int(i), gtsam.Pose2(gx, gy, 0.0), gps_model))

    # optimizer = gtsam.GaussNewtonOptimizer(graph, initial)
    # result = optimizer.optimize()
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    traj = np.array([[result.atPose2(i).x(), result.atPose2(i).y(), result.atPose2(i).theta()] for i in range(len(gt_poses))])

    return traj, gt_poses


if __name__ == "__main__":
    param_sets = [
        ([0.1, 0.1, 0.05], [0.1, 0.1, 999.0]),   
        ([0.5, 0.5, 0.25], [0.1, 0.1, 999.0]),      
        ([0.1, 0.1, 0.05], [2.5, 2.5, 999.0]),
        ([0.1, 0.1, 0.05], [4, 4, 999.0]),
    ]
    for did in range(1, 7):
        for idx, params in enumerate(param_sets):
            traj, gt = build_and_solve_pose_2d_fg(params, dataset_path=f"synthetic_pose_2d_dataset_{did}.npz")
            np.savez(f"output/pose2d_trajs/traj_ds_{did}_{idx}.npz", traj=traj, gt=gt, params=params)
            print(f"Saved output/optimized_traj_{idx}.npz with params {params}")
