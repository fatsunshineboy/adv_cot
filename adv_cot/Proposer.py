from utils.llm_answer import get_llm_answer_content


def get_discriminator_update_suggestion(discriminator_update_suggestions, instruction_discriminator, mode=0):
    """
    Generate modification suggestions based on discriminator errors
    :param mode: When mode is 0, it indicates obtaining suggestions for modifying instructions; when mode is 1, it indicates obtaining suggestions for modifying examples
    """
    if len(discriminator_update_suggestions) == 0:
        return ""
    if mode == 0:
        instruction = f"Below are examples in this format:<input>[Question]</input><output>[Answer]</output><answer>[Discriminator_Output]</answer>. In each case, the discriminator failed to detect errors in the generator’s output or identified the correct answer as incorrect. Please suggest concise, one-sentence edits to the discriminator’s instruction prompt so it more effectively detects generator output errors, focusing on common mistake patterns and how to reword the prompt to avoid them. Do not mention the specific details of the errors, names, items or data in the examples. In conclusion, the words that appear in the examples must not be included in the suggestions. Old discriminator instruction :{instruction_discriminator}. Start the output with 'New discriminator's instruction should ...'"
    elif mode == 1:
        instruction = f"Below are examples in this format:<input>[Question]</input><output>[Answer]</output><answer>[Discriminator_Output]</answer>. In each case, the discriminator failed to detect errors in the generator’s output or identified the correct answer as incorrect. In up to one sentence, provide concise, actionable suggestions for designing new discriminator examples that more effectively expose common generator mistakes. Do not mention the specific details of the errors, names, items or data in the examples. In conclusion, the words that appear in the examples must not be included in the suggestions. Discriminator instruction :{instruction_discriminator}. Start the output with 'New discriminator's examples should ...'"
    else:
        return ""
    temp = ""
    for index, example in enumerate(discriminator_update_suggestions):
        temp += f"Err{index + 1}: <input>{example['input']}</input> <output>{example['output']}</output> <answer>{example['answer']}</answer>"
    temp += "Suggestions:"
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": temp}]

    res = get_llm_answer_content(messages)

    print(f'discriminator update suggestion:\n{res}\n')
    return f"Extra suggestions can be referred to. Extra suggestions:{res}"


def get_generator_update_suggestion(generator_update_suggestions, instruction_generator, mode=0):
    """
    Generate modification suggestions based on generator errors
    :param mode: When mode is 0, it indicates obtaining suggestions for modifying instructions; when mode is 1, it indicates obtaining suggestions for modifying examples
    """
    if len(generator_update_suggestions) == 0:
        return ""
    if mode == 0:
        instruction = f"Below are examples in this format:<input>[Question]</input><output>[Answer]</output><answer>[Discriminator_Output]</answer>. In each case the discriminator marked the generator’s output as incorrect. In one sentence, give a precise, actionable modification to the generator’s instruction that targets common failure patterns to prevent similar mistakes. Do not mention the specific details of the errors, names, items or data in the examples. In conclusion, the words that appear in the examples must not be included in the suggestions. Old generator's instruction :{instruction_generator}. Start the output with 'New generator's instruction should ...'"
    elif mode == 1:
        instruction = f"Below are examples in this format:<input>[Question]</input><output>[Answer]</output><answer>[Discriminator_Output]</answer>. In each case the discriminator marked the generator’s output as incorrect. In up to one sentence, provide precise, actionable modifications to the generator’s examples that address these common failure patterns and steer its outputs away from similar errors. Do not mention the specific details of the errors, names, items or data in the examples. In conclusion, the words that appear in the examples must not be included in the suggestions. Old generator's instruction :{instruction_generator}. Start the output with 'New generator's examples should ...'"
    else:
        return ""
    temp = ""
    for index, example in enumerate(generator_update_suggestions):
        temp += f"Err{index + 1}: <input>{example['input']}</input> <output>{example['output']}</output> <answer>{example['answer']}</answer>"
    temp += "Suggestion:"
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": temp}]

    res = get_llm_answer_content(messages)
    print(f'generator update suggestion:\n{res}\n')
    return f"Extra suggestions can be referred to. Extra suggestions:{res}"