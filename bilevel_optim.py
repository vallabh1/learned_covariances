from fg import build_and_solve_pose_2d_fg
import gtsam
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import time
from pathlib import Path

def loss_function_pose2d(traj, gt):
    total = 0.0
    for i in range(len(gt)):
        # diff = gtsam.Pose2(gt[i,0], gt[i,1], gt[i,2]).between(
        #           gtsam.Pose2(traj[i,0], traj[i,1], traj[i,2]))
        diff = gtsam.Pose2(traj[i,0], traj[i,1], traj[i,2]).compose((gtsam.Pose2(gt[i,0], gt[i,1], gt[i,2]).inverse()))
        vec = gtsam.Pose2.Logmap(diff)
        total += np.dot(vec, vec)
    return total / len(gt)

def residual_vector_pose2d(traj, gt):
    vecs = []
    for i in range(len(gt)):
        diff = gtsam.Pose2(gt[i,0], gt[i,1], gt[i,2]).between(
                   gtsam.Pose2(traj[i,0], traj[i,1], traj[i,2]))
        v = gtsam.Pose2.Logmap(diff)
        vecs.append(v)
    return np.concatenate(vecs)   # shape (3N,)

def _compute_jacobian_column(args):
    """Helper function to compute a single column of the Jacobian matrix."""
    i, theta, eps, dataset_path, base_vec, gt = args
    perturbed = theta.copy()
    perturbed[i] += eps
    traj_p, _ = build_and_solve_pose_2d_fg((perturbed[0:3], perturbed[3:6]), dataset_path)
    vec_p = residual_vector_pose2d(traj_p, gt)
    return i, (vec_p - base_vec) / eps

def getJacobian_pose2d(theta, eps=1e-3, dataset_path="synthetic_pose_2d_dataset_1.npz", parallel=False, n_workers=8):
    base_traj, gt = build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path)
    base_vec = residual_vector_pose2d(base_traj, gt)
    # print(parallel)

    m = len(theta)
    n = len(base_vec)
    S = np.zeros((n, m))

    if parallel:
        if n_workers is None:
            n_workers = min(cpu_count(), m)
        # print(f"Using {n_workers} workers")
        
        args_list = [(i, theta, eps, dataset_path, base_vec, gt) for i in range(m)]
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_compute_jacobian_column, args_list))
        
        for i, column in results:
            S[:, i] = column
    else:
        # print("here")
        for i in range(m):
            # start_time = time.time()
            perturbed = theta.copy()
            perturbed[i] += eps
            traj_p, _ = build_and_solve_pose_2d_fg((perturbed[0:3], perturbed[3:6]), dataset_path)
            vec_p = residual_vector_pose2d(traj_p, gt)
            S[:, i] = (vec_p - base_vec) / eps
            # end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")

    return S, base_vec


def _dataset_grad_contribution(args):
    dataset_id, theta, eps, dataset_path_template = args
    dataset_path = dataset_path_template.format(did=dataset_id)
    # Disable per-theta parallelism here to avoid nested parallelism
    S, res = getJacobian_pose2d(theta, eps, dataset_path, parallel=True)
    return S.T @ res / len(res)

def gradient_pose2d(theta, eps=1e-3, dataset_path_template="synthetic_pose_2d_dataset_{did}.npz", dataset_size=6):
    grad = np.zeros_like(theta)
    args = [(dataset_id, theta, eps, dataset_path_template) for dataset_id in range(1, dataset_size+1)]
    # Parallelize across datasets using processes
    with ProcessPoolExecutor(max_workers=min(cpu_count(), dataset_size)) as executor:
        contributions = list(executor.map(_dataset_grad_contribution, args))
    for contrib in contributions:
        grad += contrib
    grad /= dataset_size
    return grad



def frank_wolfe_step(theta, grad, t, lam_min, lam_max, M=1000):
    """
    Perform one Frank-Wolfe step with box constraints.
    theta: current parameter vector
    grad: gradient vector (∂L/∂θ)
    t: iteration index
    lam_min, lam_max: arrays of same shape as theta
    M: parameter controlling step size decay
    """
    # LP over box constraints
    s_star = np.zeros_like(theta)
    for i in range(len(theta)):
        if grad[i] > 0:
            s_star[i] = lam_min[i]
        else:
            s_star[i] = lam_max[i]

    alpha = 2.0 / (M + t + 1)

    # Update rule
    return theta + alpha * (s_star - theta)

def bilevel_optim_pose_2d_frank_wolfe(theta_initial, lam_min, lam_max, eps=1e-6, dataset_path_template="synthetic_pose_2d_dataset_{did}.npz", dataset_size=6):
    theta = theta_initial.copy()
    loss_graph = []
    best_theta = theta.copy()
    best_loss = float(0)
    for dataset_id in range(1, dataset_size+1):
        dataset_path = dataset_path_template.format(did=dataset_id)
        best_loss += loss_function_pose2d(*build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path))
    best_loss /= dataset_size
    loss_graph.append(best_loss)
    # todo: implement early stopping w patience with validation trajectories
    for i in range(2000):  
        grad = gradient_pose2d(theta, eps, dataset_path_template, dataset_size)
        theta = frank_wolfe_step(theta, grad, i, lam_min, lam_max, M=2000)
        loss = float(0)
        for dataset_id in range(1, dataset_size+1):
            dataset_path = dataset_path_template.format(did=dataset_id)
            loss += loss_function_pose2d(*build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path))
        loss /= dataset_size
        loss_graph.append(loss)

        if loss < best_loss:
            best_theta = theta.copy()
            best_loss = loss

        if i % 10 == 0:
            print(f"Iter {i}: Loss={loss:.4f}, theta={theta}")

    return best_theta, loss_graph


def optimize_params(theta_initial, eps=1e-6, dataset_path_template="synthetic_pose_2d_dataset_{did}.npz", dataset_size=6):
    lam_min = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 500])
    lam_max = np.array([3, 3, 3, 10, 10, 9999])
    # theta, loss_graph = bilevel_optim_pose2d(theta_initial, eps, learning_rate, dataset_path)
    theta, loss_graph = bilevel_optim_pose_2d_frank_wolfe(theta_initial, lam_min, lam_max, eps, dataset_path_template, dataset_size)
    for did in range(1, dataset_size+1):
        dataset_path = dataset_path_template.format(did=did)
        optimized_traj, gt = build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path)
        np.savez(f"output/pose2d_trajs/optim_traj_ds_{dataset_path.split('_')[4][0]}.npz", traj=optimized_traj, gt=gt, params=theta_initial)
        print(theta[0], theta[1], theta[2])
        print(theta[3], theta[4], theta[5])
        print(len(loss_graph))
        plt.plot(loss_graph)
        # plt.show()
        plt.savefig(f"output/figs/bilevel_optim_loss.png")
        plt.close()


if __name__ == "__main__":
    theta_initial = np.array([0.02, 0.02, 0.02,3, 3, 999.0])
    optimize_params(theta_initial, dataset_path_template="synthetic_pose_2d_dataset_{did}.npz", dataset_size=6)