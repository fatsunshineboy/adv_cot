from adv_cot.Modifier import format_messages
from adv_cot.Verifier import verify
from utils.extract_answer import evaluate_answer
from utils.save_res import save_err_to_file


def get_accuracy(test_samples, prompt,GLOBAL_DATA,dir_path):
    # Record the number of correct answers
    score = 0
    # Record the total number of problems
    number_count = 0

    instruction_generator = prompt["instruction_generator"]
    examples_generator = prompt["examples_generator"]
    instruction_discriminator = prompt["instruction_discriminator"]
    examples_discriminator = prompt["examples_discriminator"]

    generator_prompt = format_messages(instruction_generator, examples_generator[:])
    discriminator_prompt = format_messages(instruction_discriminator, examples_discriminator[:])
    for test_sample in test_samples:
        number_count += 1
        sample_id = test_sample["id"]
        sample_input = test_sample["input"]
        sample_output = test_sample["output"][0]
        generator_output = verify(generator_prompt[:], sample_input, discriminator_prompt[:],GLOBAL_DATA)
        evaluate_result = evaluate_answer(generator_output, sample_input, GLOBAL_DATA.get("dataset_name"), sample_output)

        if evaluate_result:
            score += 1
        else:
            save_err_to_file(sample_id, sample_input, generator_output, dir_path)
        print(f'number:{number_count} score:{score}\n')
    accuracy = (score / number_count) * 100
    return accuracy, score, number_count