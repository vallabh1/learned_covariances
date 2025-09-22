import numpy as np
import matplotlib.pyplot as plt

def plot_pose2d_trajectories(file_list):
    plt.figure(figsize=(8, 6))

    gt_data = np.load(f"output/{file_list[0]}", allow_pickle=True)
    gt = gt_data["gt"]

    plt.plot(gt[:,0], gt[:,1], "k-", label="Ground Truth", linewidth=2)

    for f in file_list:
        data = np.load(f"output/{f}", allow_pickle=True)
        traj = data["traj"]
        params = data["params"]

        label = f"odom={params[0]}, gps={params[1]}"
        if f == "bilevel_optim_traj.npz":
            label = f"Bilevel Optim"
        
        plt.plot(traj[:,0], traj[:,1], marker="o", markersize=3, label=label)
        

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend(loc='lower right', fontsize=8)
    plt.title("Optimized Trajectories vs Ground Truth")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(f'output/figs/pose2d_trajectories.png')
    plt.close()


if __name__ == "__main__":
    files = [
        "optimized_traj_0.npz",
        "optimized_traj_1.npz",
        "optimized_traj_3.npz",
        "bilevel_optim_traj.npz"]

    plot_pose2d_trajectories(files)
