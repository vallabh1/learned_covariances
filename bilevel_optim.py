from fg import build_and_solve_pose_2d_fg
import gtsam
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import time
def loss_function_pose2d(traj, gt):
    total = 0.0
    for i in range(len(gt)):
        diff = gtsam.Pose2(gt[i,0], gt[i,1], gt[i,2]).between(
                  gtsam.Pose2(traj[i,0], traj[i,1], traj[i,2]))
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

def getJacobian_pose2d(theta, eps=1e-3, dataset_path="synthetic_pose_2d_dataset_1.npz", parallel=True, n_workers=16):
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
        
        # Prepare arguments for parallel computation
        args_list = [(i, theta, eps, dataset_path, base_vec, gt) for i in range(m)]
        
        # Use ThreadPoolExecutor for I/O bound tasks (file loading)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_compute_jacobian_column, args_list))
        
        # Fill the Jacobian matrix
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

def numerical_gradient_pose2d(theta, eps=1e-3, dataset_path="synthetic_pose_2d_dataset_1.npz"):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_p = theta.copy(); theta_p[i] += eps
        theta_m = theta.copy(); theta_m[i] -= eps
        Lp = loss_function_pose2d(*build_and_solve_pose_2d_fg((theta_p[0:3], theta_p[3:6]), dataset_path))
        Lm = loss_function_pose2d(*build_and_solve_pose_2d_fg((theta_m[0:3], theta_m[3:6]), dataset_path))
        grad[i] = (Lp - Lm) / (2*eps)
    return grad

def gradient_pose2d(theta, eps=1e-3, dataset_path="synthetic_pose_2d_dataset_1.npz"):
    S, res = getJacobian_pose2d(theta, eps, dataset_path, parallel=True, n_workers=24)
    grad = S.T @ res / len(res)
    return grad



def bilevel_optim_numerical_grad_pose2d(theta_initial, eps=1e-6, learning_rate=1e-6, dataset_path="synthetic_pose_2d_dataset_1.npz"):
    theta = theta_initial.copy()
    loss_and_params = []
    best_theta = theta.copy()
    best_loss = loss_function_pose2d(*build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path))
    loss_and_params.append(loss_function_pose2d(*build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path)))
    for i in range(10000):
        
        grad = numerical_gradient_pose2d(theta, eps)
        print(i, theta, grad) if i % 10 == 0 else None
        theta -= learning_rate * grad
        loss = loss_function_pose2d(*build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path))
        loss_and_params.append(loss)
        if loss < best_loss:
            best_theta = theta.copy()
            best_loss = loss
        
    return best_theta, loss_and_params

def bilevel_optim_pose2d(theta_initial, eps=1e-6, learning_rate=1e-5, dataset_path="synthetic_pose_2d_dataset_1.npz"):
    theta = theta_initial.copy()
    loss_and_params = []
    best_theta = theta.copy()
    best_loss = loss_function_pose2d(*build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path))
    loss_and_params.append(loss_function_pose2d(*build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path)))
    for i in range(2500):
        grad = gradient_pose2d(theta,eps, dataset_path)
        print(i, theta, grad) if i % 10 == 0 else None
        theta -= learning_rate * grad
        loss = loss_function_pose2d(*build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path))
        loss_and_params.append(loss)
        if loss < best_loss:
            best_theta = theta.copy()
            best_loss = loss
    return best_theta, loss_and_params

def optimize_params(theta_initial, eps=1e-6, learning_rate=2e-4, dataset_path="synthetic_pose_2d_dataset_1.npz"):
    if dataset_path == "synthetic_pose_2d_dataset_1.npz":
        learning_rate = np.array([2e-4, 2e-4, 2e-4, 1e-3, 1e-3, 1e-7])

        theta, loss_and_params = bilevel_optim_pose2d(theta_initial, eps, learning_rate, dataset_path)
        optimized_traj, gt = build_and_solve_pose_2d_fg((theta[0:3], theta[3:6]), dataset_path)
        np.savez(f"output/bilevel_optim_traj.npz", traj=optimized_traj, gt=gt, params=theta_initial)
        print(theta[0], theta[1], theta[2])
        print(theta[3], theta[4], theta[5])
        print(len(loss_and_params))
        plt.plot(loss_and_params)
        # plt.show()
        plt.savefig(f"output/figs/bilevel_optim_loss.png")
        plt.close()

if __name__ == "__main__":
    theta_initial = np.array([0.05, 0.02, 0.02,2.2, 2.2, 999.0])
    optimize_params(theta_initial)