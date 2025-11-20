import os
import random
import json

from adv_cot.Discriminator import compute_loss
from adv_cot.Modifier import format_messages, update_discriminator, update_generator
from adv_cot.accuracy import get_accuracy
from utils.save_res import save_result_to_file


def run(task_name,GLOBAL_DATA,dir_path):
    """
    Calculate the optimal prompt and accuracy of the current task
    :param task_name: teh task name that need training
    :return: accuracy that under multiple iterations
    """
    # num of train set extracted
    num_train_instances = 100
    # load json

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TASK_DIR = os.path.join(BASE_DIR, "tasks")

    with open(os.path.join(TASK_DIR, f"{task_name}.json"), "r", encoding="UTF-8") as f:
        data = json.load(f)

    # Generator's instruction
    instruction_generator = data.get('Instruction_Generator', "")
    # Discriminator's instruction
    instruction_discriminator = data.get('Instruction_Discriminator', "")
    # Generator's examples
    examples_generator = data.get('Examples_Generator', [])
    # Discriminator's examples
    examples_discriminator = data.get('Examples_Discriminator', [])
    # test datasets
    instances = data.get('Instances', [])
    # extract train set for discriminator from test datasets ( only input )
    train_samples = random.sample(instances, num_train_instances)
    # begin optimize
    optimize_res_arr, loss_count = optimize(instruction_generator, examples_generator[:], instruction_discriminator,
                                            examples_discriminator[:],
                                            train_samples[:],GLOBAL_DATA)

    save_result_to_file(f"adv_cot-{task_name}-optimize_res", optimize_res_arr, dir_path)
    save_result_to_file(f"adv_cot-{task_name}-loss_count", f"loss_count:{loss_count}", dir_path)
    save_result_to_file(f"adv_cot-{task_name}-GLOBAL_DATA", f"GLOBAL_DATA:{json.dumps(GLOBAL_DATA)}",
                        dir_path)

    if "coin" in task_name:
        test_samples = random.sample(instances, 500)
    else:
        test_samples = instances

    test_accuracy_arr = []
    for index, optimize_prompt in enumerate(optimize_res_arr):
        print(f"\n---------------------compute accuracy {index + 1}---------------------\n")
        accuracy, score, number_count = get_accuracy(test_samples, optimize_prompt,GLOBAL_DATA,dir_path)
        test_accuracy_arr.append(accuracy)
        print(f"Time {index + 1} accuracy:{accuracy}\n")
        save_result_to_file(f"adv_cot-{task_name}-optimize_res-Time {index + 1} accuracy:",
                            f"accuracy:{accuracy} score:{score} num_count:{number_count}",
                            dir_path)

    return test_accuracy_arr


def optimize(instruction_generator, examples_generator_full, instruction_discriminator, examples_discriminator_full,
             train_samples_full,GLOBAL_DATA):
    """
    optimize generator and discriminator
    :return: optimized set of prompt
    """

    # num of extract examples randomly
    num_examples_extracted = GLOBAL_DATA.get("num_examples_extracted")
    # num of iteration
    num_iterate_optimize = GLOBAL_DATA.get("num_iterate_optimize")

    # Randomly select examples as the initial of the generator
    examples_generator = random.sample(examples_generator_full, num_examples_extracted)
    # Randomly select examples as the initial of the discriminator
    examples_discriminator = random.sample(examples_discriminator_full, num_examples_extracted)

    optimize_res_arr = []
    # initial loss
    loss = 0
    generator_update_suggestions = []
    discriminator_update_suggestions = []
    loss_count = 0

    for iterate_count in range(num_iterate_optimize):

        # Calculate the changes in loss
        loss_arr = []

        # Randomly select as training sets to prevent overfitting
        true_samples = random.sample(examples_generator_full, num_examples_extracted)
        train_samples = random.sample(train_samples_full, num_examples_extracted)

        print(f"\n--------------------Start optimize {iterate_count} --------------------\n")
        print(f"Before optimize:\n")
        print(f"instruction_generator:\n{instruction_generator}\n")
        print(f"examples_generator:\n{"\n".join(str(example) for example in examples_generator)}\n")
        print(f"instruction_discriminator:\n{instruction_discriminator}\n")
        print(f"examples_discriminator:\n{"\n".join(str(example) for example in examples_discriminator)}\n")

        # Construct the generator prompt
        generator_prompt = format_messages(instruction_generator, examples_generator)
        # Construct the discriminator prompt
        discriminator_prompt = format_messages(instruction_discriminator, examples_discriminator)

        if loss == 0:
            # Calculate the initial loss
            loss, generator_update_suggestions, discriminator_update_suggestions = compute_loss(
                generator_prompt[:], discriminator_prompt[:], true_samples[:], train_samples[:])
            loss_arr.append(loss)
        print(f"loss:\n{loss}\n")
        # Update the discriminator
        instruction_discriminator, examples_discriminator, loss, generator_update_suggestions, discriminator_update_suggestions = update_discriminator(
            generator_prompt[:],
            instruction_discriminator,
            examples_discriminator[:],
            true_samples[:],
            train_samples[:],
            loss,
            generator_update_suggestions[:],
            discriminator_update_suggestions[
            :], loss_arr,GLOBAL_DATA)
        print(f"After discriminator optimize:\n")
        print(f"loss:\n{loss}\n")
        print(f"instruction_discriminator:\n{instruction_discriminator}\n")
        print(f"examples_discriminator:\n{"\n".join(str(example) for example in examples_discriminator)}\n")

        # Update the discriminator prompt
        discriminator_prompt = format_messages(instruction_discriminator, examples_discriminator)
        # Update the generator
        instruction_generator, examples_generator, loss, generator_update_suggestions, discriminator_update_suggestions = update_generator(
            instruction_generator, examples_generator[:],
            discriminator_prompt[:], true_samples[:],
            train_samples[:], loss,
            generator_update_suggestions[:],
            discriminator_update_suggestions[
            :], loss_arr,GLOBAL_DATA)

        print(f"After generator optimize:\n")
        print(f"instruction_generator:\n{instruction_generator}\n")
        print(f"examples_generator:\n{"\n".join(str(example) for example in examples_generator)}\n")
        print(f"instruction_discriminator:\n{instruction_discriminator}\n")
        print(f"examples_discriminator:\n{"\n".join(str(example) for example in examples_discriminator)}\n")
        print(f"loss:\n{loss}\n")
        print(f"loss change:\n{loss_arr}\n")
        print("\n--------------------End optimize--------------------\n")
        # Save when the loss value is greater than -2 * num_examples_extracted
        # if loss > (-2) * num_examples_extracted:
        rounded_loss = round(loss, 6)
        # Prevent the save of duplicate losses
        if not any(round(item["loss"], 6) == rounded_loss for item in optimize_res_arr):
            optimize_res_arr.append({
                "instruction_generator": instruction_generator,
                "examples_generator": examples_generator,
                "instruction_discriminator": instruction_discriminator,
                "examples_discriminator": examples_discriminator,
                "loss": loss,
                "index": iterate_count
            })

        loss_count += len(loss_arr)

    return optimize_res_arr, loss_count