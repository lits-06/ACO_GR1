from oss import *
import numpy as np
from tqdm import tqdm
import json
from statistics import mean

num_tasks = dimension
sol = [None] * num_ants
sol_cost = np.zeros(num_ants)
best_so_far = None
best_cost = 1e300
ibest = num_tasks

ALPHA = 1
BETA = 1
tau0 = 0.5
rho = 0.05
w_bs = 0.2
w_ib = 0.6
phero = np.ones((num_tasks + 1, num_tasks)) * tau0


def aco():
    results_control = {}
    # np.random.seed(seed)
    for i in tqdm(range(1, num_iterations + 1)):

        this_cycle_times = []  #

        for ant_num in range(0, num_ants):
            sol[ant_num] = generate_solution()

            path_time, _ = objective_function(sol[ant_num])
            this_cycle_times.append(path_time)

        evaluate_solutions()
        update_pheromone()

        results_control.update(
            {i: [min(this_cycle_times), mean(this_cycle_times), max(this_cycle_times)]}
        )

        json.dump(
            results_control, open("results/ACO/json/ACO_cycles_results.json", "w")
        )

    return best_so_far, best_cost


def generate_solution():
    sol = []
    task_to_process = list(range(num_tasks))
    current_task = num_tasks  # a non existing task, it means that it must choose the first task to process
    for i in range(num_tasks):
        next_task = select_task(current_task, task_to_process)
        sol.append(next_task)
        current_task = next_task
        task_to_process.remove(next_task)

    return sol


def select_task(current_task, task_to_process):
    num_task_to_process = len(task_to_process)  # num task remaining to process
    prob = np.zeros(num_task_to_process)
    for i in range(0, num_task_to_process):
        task = task_to_process[i]
        tau = phero[current_task][task]  # pheromone value
        eta = (
            1
            if current_task == num_tasks
            else 1 / processing_times[task // num_machines][task % num_machines]
        )  # heuristic value
        p_i_to_j = (
            tau**ALPHA * eta**BETA
        )  # Probability of choosing task j as next task in solution path
        prob[i] = p_i_to_j

    # Normalize probabilities to sum 1 for numpy randon choice method
    den = sum(prob)
    normalized_probabilities = prob / den
    indexs = [i for i in range(len(normalized_probabilities))]
    next_node_index = np.random.choice(
        indexs, p=normalized_probabilities
    )  # Roulette Wheel method
    return task_to_process[next_node_index]


def evaporate():
    global phero
    phero = (1 - rho) * phero


def evaluate_solutions():
    global best_cost, best_so_far, ibest

    for i in range(0, num_ants):
        sol_cost[i], _ = objective_function(
            sol[i]
        )  # Save solution cost for each pariticle

    ibest = sol_cost.argmin()  # Take best solution (minimum cost) for the pariticle
    if (
        sol_cost[ibest] < best_cost
    ):  # If the iteration best is better than best so far replace the solution
        best_cost = sol_cost[ibest]
        best_so_far = sol[ibest]


def update_pheromone():
    evaporate()
    reward(best_so_far, w_bs)
    reward(sol[ibest], w_ib)


def reward(sol, delta):
    global phero
    current = num_tasks
    for i in range(num_tasks):
        task = sol[i]
        phero[current][task] += delta
        current = task
