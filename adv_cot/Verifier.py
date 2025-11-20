from adv_cot.Discriminator import get_discriminator_output
from adv_cot.Generator import get_generator_output


def verify(generator_prompt, sample, discriminator_prompt,GLOBAL_DATA):
    """
    Take the one with the highest confidence level as the answer
    """
    num_iterate_verify = GLOBAL_DATA.get("num_iterate_verify")
    logprobs_arr = []
    for i in range(num_iterate_verify):
        generator_output = get_generator_output(generator_prompt[:], sample)
        log_probability, *arg = get_discriminator_output(discriminator_prompt[:],
                                                         {"input": sample,
                                                          "output": generator_output})
        logprobs_arr.append({"logprobs": log_probability, "generator_output": generator_output})

    for log_item in logprobs_arr:
        print(f"log_item:{log_item}")

    return max(logprobs_arr, key=lambda x: x["logprobs"])["generator_output"]