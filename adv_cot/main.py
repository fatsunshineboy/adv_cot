import datetime
import os
import sys
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from adv_cot.optimize_prompt import run
from utils.task_list import get_task_list

# Define global variables
GLOBAL_DATA = {
    "version": 1.0,
    # the number of candidate instruction and examples
    "num_iterate_max": 10,
    # the number of examples extracted in prompt
    "num_examples_extracted": 5,
    # the number of optimized rounds
    "num_iterate_optimize": 3,
    # the number of verification iterations when obtaining accuracy
    "num_iterate_verify": 3,
    # Self‑Consistency
    "num_samples": 10,
    "dataset_name":""
}


def main(task):
    GLOBAL_DATA["dataset_name"] = task
    if "letter" in task:
        GLOBAL_DATA["num_examples_extracted"] = 4
    else:
        GLOBAL_DATA["num_examples_extracted"] = 5

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    dir_path = Path.cwd() / "log_adv_cot"/ task / f"{task}-{current_time}"

    print(dir_path)
    os.makedirs(dir_path, exist_ok=True)

    # f = open(f"{dir_path}/console.log", "w", encoding="UTF-8")

    # sys.stdout = f

    try:
        run(task,GLOBAL_DATA,dir_path)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    task_list = get_task_list()

    for task_name in task_list:
        main(task_name)

    # with ProcessPoolExecutor() as executor:
    #     executor.map(main, task_list)