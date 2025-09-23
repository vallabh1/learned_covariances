import numpy as np
import matplotlib.pyplot as plt

def plot_pose2d_trajectories(file_list, dataset_id):
    plt.figure(figsize=(8, 6))

    gt_data = np.load(f"output/pose2d_trajs/{file_list[0]}", allow_pickle=True)
    gt = gt_data["gt"]

    plt.plot(gt[:,0], gt[:,1], "k-", label="Ground Truth", linewidth=2)

    for f in file_list:
        data = np.load(f"output/pose2d_trajs/{f}", allow_pickle=True)
        traj = data["traj"]
        params = data["params"]

        label = f"odom={params[0]}, gps={params[1]}"
        if f == f"optim_traj_ds_{dataset_id}.npz":
            label = f"Bilevel Optim"
        
        plt.plot(traj[:,0], traj[:,1], marker="o", markersize=3, label=label)
        

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("Optimized Trajectories vs Ground Truth")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(f'output/figs/pose2d_trajs_ds_{dataset_id}.png')
    plt.close()


if __name__ == "__main__":
    dataset_id = 1
    # files = [
    #     "traj_ds_{dataset_id}_0.npz",
    #     "traj_ds_{dataset_id}_1.npz",
    #     "traj_ds_{dataset_id}_2.npz",
    #     "traj_ds_{dataset_id}_3.npz",
    #     "optim_traj_ds_{dataset_id}.npz"]

    for did in range(1, 6):
        files = [
            # f"traj_ds_{did}_0.npz",
            # f"traj_ds_{did}_1.npz",
            # f"traj_ds_{did}_2.npz",
            f"traj_ds_{did}_3.npz",
            f"optim_traj_ds_{did}.npz"]
        plot_pose2d_trajectories(files, did)
