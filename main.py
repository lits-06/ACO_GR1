from aco import *


solution_schedule, solution_cost = aco()
_, schedule = objective_function(solution_schedule)
results = get_gantt(schedule)
visualize(results, "ACO")
