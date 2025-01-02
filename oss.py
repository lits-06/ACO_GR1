import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import json
import re
import sys

num_ants = 10
num_iterations = 500

dataset_folder = "dataset"
file_name = "ta01"

if len(sys.argv) > 1:
    file_name = sys.argv[1]

full_path = os.path.join(dataset_folder, file_name)

with open(full_path) as f:
    lines = f.readlines()

first_line = lines[1].split()
num_jobs = int(first_line[0])
num_machines = int(first_line[1])

processing_times_task_order = [
    [int(lines[i].split()[j]) for j in range(num_machines)]
    for i in range(3, 3 + num_jobs)
]

machine_index = [
    [int(lines[i].split()[j]) - 1 for j in range(num_machines)]
    for i in range(4 + num_jobs, 4 + 2 * num_jobs)
]

# processing_times_task_order = [
#     [94, 66, 10, 53, 26, 15, 65, 82, 10, 27, 93, 92, 96, 70, 83],
#     [74, 31, 88, 51, 57, 78, 8, 7, 91, 79, 18, 51, 18, 99, 33],
#     [4, 82, 40, 86, 50, 54, 21, 6, 54, 68, 82, 20, 39, 35, 68],
#     [73, 23, 30, 30, 53, 94, 58, 93, 32, 91, 30, 56, 27, 92, 9],
#     [78, 23, 21, 60, 36, 29, 95, 99, 79, 76, 93, 42, 52, 42, 96],
#     [29, 61, 88, 70, 16, 31, 65, 83, 78, 26, 50, 87, 62, 14, 30],
#     [18, 75, 20, 4, 91, 68, 19, 54, 85, 73, 43, 24, 37, 87, 66],
#     [32, 52, 9, 49, 61, 35, 99, 62, 6, 62, 7, 80, 3, 57, 7],
#     [85, 30, 96, 91, 13, 87, 82, 83, 78, 56, 85, 8, 66, 88, 15],
#     [5, 59, 30, 60, 41, 17, 66, 89, 78, 88, 69, 45, 82, 6, 13],
#     [90, 27, 1, 8, 91, 80, 89, 49, 32, 28, 90, 93, 6, 35, 73],
#     [47, 43, 75, 8, 51, 3, 84, 34, 28, 60, 69, 45, 67, 58, 87],
#     [65, 62, 97, 20, 31, 33, 33, 77, 50, 80, 48, 90, 75, 96, 44],
#     [28, 21, 51, 75, 17, 89, 59, 56, 63, 18, 17, 30, 16, 7, 35],
#     [57, 16, 42, 34, 37, 26, 68, 73, 5, 8, 12, 87, 83, 20, 97],
# ]


# num_jobs = len(processing_times_task_order)
# num_machines = len(processing_times_task_order[0])
dimension = num_jobs * num_machines


# machines = [
#     [7, 13, 5, 8, 4, 3, 11, 12, 9, 15, 10, 14, 6, 1, 2],
#     [5, 6, 8, 15, 14, 9, 12, 10, 7, 11, 1, 4, 13, 2, 3],
#     [2, 9, 10, 13, 7, 12, 14, 6, 1, 3, 8, 11, 5, 4, 15],
#     [6, 3, 10, 7, 11, 1, 14, 5, 8, 15, 12, 9, 13, 2, 4],
#     [8, 9, 7, 11, 5, 10, 3, 15, 13, 6, 2, 14, 12, 1, 4],
#     [6, 4, 13, 14, 12, 5, 15, 8, 3, 2, 11, 1, 10, 7, 9],
#     [13, 4, 8, 9, 15, 7, 2, 12, 5, 6, 3, 11, 1, 14, 10],
#     [12, 6, 1, 8, 13, 14, 15, 2, 3, 9, 5, 4, 10, 7, 11],
#     [11, 12, 7, 15, 1, 2, 3, 6, 13, 5, 9, 8, 10, 14, 4],
#     [7, 12, 10, 3, 9, 1, 14, 4, 11, 8, 2, 13, 15, 5, 6],
#     [5, 8, 14, 1, 6, 13, 7, 9, 15, 11, 4, 2, 12, 10, 3],
#     [3, 15, 1, 13, 7, 11, 8, 6, 9, 10, 14, 2, 4, 12, 5],
#     [6, 9, 11, 3, 4, 7, 10, 1, 14, 5, 2, 12, 13, 8, 15],
#     [9, 15, 5, 14, 6, 7, 10, 2, 13, 8, 12, 11, 4, 3, 1],
#     [11, 9, 13, 7, 5, 2, 14, 15, 12, 1, 8, 4, 3, 10, 6],
# ]

# Reorder processing times: processingTime[j][m] is the processing time of the
# task of job j that is processed on machine m
processing_times = [
    [
        processing_times_task_order[j][machine_index[j].index(m)]
        for m in range(num_machines)
    ]
    for j in range(num_jobs)
]


def objective_function(solution):
    schedule = (
        []
    )  # Contains n list (n = num machines) with a scheduling for each machine
    for i in range(num_machines):
        schedule.append([])

    completed = [0] * num_jobs  # index in machines
    time_end_job = [0] * num_jobs

    for task in solution:  # for each task in solution path
        job = task % num_jobs
        machine = machine_index[job][completed[job]]

        execution_time = processing_times[job][machine]

        start_time = find_start_time(
            schedule, time_end_job, job, machine, execution_time
        )
        if start_time < len(schedule[machine]):  # time end machine
            for t in range(start_time, start_time + execution_time):
                schedule[machine][t] = job

            time_end_job[job] = start_time + execution_time

        else:
            if len(schedule[machine]) < start_time:
                schedule[machine].extend(["-"] * (start_time - len(schedule[machine])))

            schedule[machine].extend([job] * execution_time)
            time_end_job[job] = len(schedule[machine])

        completed[job] += 1

    machine_execution_lengths = []
    for i in range(num_machines):  # calculate makespan
        machine_execution_lengths.append(len(schedule[i]))

    return max(machine_execution_lengths), schedule


def find_start_time(schedule, time_end_job, job, machine, execution_time):
    start_time = time_end_job[job]
    end_time = len(schedule[machine])

    for t in range(start_time, end_time - execution_time + 1):
        if all(slot == "-" for slot in schedule[machine][t : t + execution_time]):
            return t

    return max(start_time, end_time)


def get_gantt(solution):
    gantt_data = []  # machine list of dict
    job_json_data = {}
    for i in range(num_machines):
        schedule = {}
        machine_num = i
        time = 0
        job_num = solution[i][0]
        start = time
        duration = 0
        for j in range(len(solution[i])):
            if job_num == solution[i][j]:
                duration += 1
                time += 1
            else:
                schedule = get_dict(job_num, machine_num, start, duration)
                gantt_data.append(schedule)
                start = time
                time += 1
                job_num = solution[i][j]
                duration = 1

        schedule = get_dict(
            job_num, machine_num, start, duration
        )  # Do it because the last control cannot be done (out of bounds with index)
        gantt_data.append(schedule)

    gantt_data = remove_null_execution(gantt_data)

    for key in gantt_data:
        if key["Job"] not in job_json_data:
            job_json_data[key["Job"]] = []

        job_json_data[key["Job"]].append(
            {
                "Machine": key["Machine"],
                "Start": key["Start"],
                "Duration": key["Duration"],
                "Finish": key["Finish"],
            }
        )

    job_json_data_sorted = {
        key: job_json_data[key]
        for key in sorted(job_json_data.keys(), key=extract_number)
    }

    for job_key, job_list in job_json_data_sorted.items():
        job_list.sort(key=lambda x: x["Start"])

    json.dump(job_json_data_sorted, open("results/ACO/json/job_results.json", "w"))

    return gantt_data


def get_dict(job_num, machine_num, start, duration):
    if job_num != "-":
        job_num += 1
        machine_num += 1

    return {
        "Job": "job_" + str(job_num),
        "Machine": "machine_" + str(machine_num),
        "Start": start,
        "Duration": duration,
        "Finish": start + duration,
    }


def remove_null_execution(scheduling):
    new_scheduling = []
    for sched in scheduling:
        if sched["Job"] == "job_-":
            pass
        else:
            new_scheduling.append(sched)

    return new_scheduling


def extract_number(job):
    return int(re.search(r"\d+", job).group())


def visualize(results, algo):
    schedule = pd.DataFrame(results)
    JOBS = sorted(list(schedule["Job"].unique()))
    MACHINES = sorted(list(schedule["Machine"].unique()))
    makespan = schedule["Finish"].max()

    bar_style = {"alpha": 1.0, "lw": 25, "solid_capstyle": "butt"}
    text_style = {"color": "white", "weight": "bold", "ha": "center", "va": "center"}
    colors = mpl.cm.Dark2.colors

    schedule.sort_values(by=["Job", "Start"])
    schedule.set_index(["Job", "Machine"], inplace=True)

    fig, ax = plt.subplots(2, 1, figsize=(12, 5 + (len(JOBS) + len(MACHINES)) / 4))

    for jdx, j in enumerate(JOBS, 1):
        for mdx, m in enumerate(MACHINES, 1):
            if (j, m) in schedule.index:
                xs = schedule.loc[(j, m), "Start"]
                xf = schedule.loc[(j, m), "Finish"]
                ax[0].plot([xs, xf], [jdx] * 2, c=colors[mdx % 7], **bar_style)
                ax[0].text((xs + xf) / 2, jdx, m, **text_style)
                ax[1].plot([xs, xf], [mdx] * 2, c=colors[jdx % 7], **bar_style)
                ax[1].text((xs + xf) / 2, mdx, j, **text_style)

    ax[0].set_title("Jobs Schedule")
    ax[0].set_ylabel("Jobs")
    ax[1].set_title("Machines Schedule")
    ax[1].set_ylabel("Machines")

    for idx, s in enumerate([JOBS, MACHINES]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        ax[idx].set_yticklabels(s)
        ax[idx].text(
            makespan,
            ax[idx].get_ylim()[0] - 0.6,
            "{0:0.1f}".format(makespan),
            ha="center",
            va="top",
        )
        ax[idx].plot([makespan] * 2, ax[idx].get_ylim(), "r--")
        ax[idx].set_xlabel("Time")
        ax[idx].grid(True)

    fig.tight_layout()
    plt.plot()

    folder_path = f"results/{algo}/images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith((".png"))]
    num_images = len(image_files)
    image_name = f"{folder_path}/execution_gantt_{num_images + 1}.png"
    plt.savefig(image_name)
