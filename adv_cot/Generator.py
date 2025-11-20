from utils.llm_answer import get_llm_answer_content


def get_generator_output(generator_prompt, sample):
    """
    The output of the generator
    """
    generator_prompt.append({"role": "user",
                             "content": sample + "End the output with the answer is"})
    generator_output = get_llm_answer_content(generator_prompt, 0.6,seed_flag=False)
    return generator_output