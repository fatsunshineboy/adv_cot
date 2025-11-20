from adv_cot.Discriminator import compute_loss
from adv_cot.Proposer import get_discriminator_update_suggestion, get_generator_update_suggestion
from utils.llm_answer import get_llm_answer_content

import re

def update_discriminator(generator_prompt, instruction_discriminator_old,
                         examples_discriminator_old, true_samples, train_samples, loss_old,
                         generator_update_suggestions_old,
                         discriminator_update_suggestions_old, loss_arr,GLOBAL_DATA):
    """
    Update the instructions and examples of the discriminator
    """
    # The number of candidate instruction and examples
    num_iterate_max = GLOBAL_DATA.get("num_iterate_max")
    # the instruction of updating instruction
    update_dis_ins_instruction_ori = "Try to generate a new instruction to make LLM discriminator more precise to find LLM generator's errors. Keep the task instruction as declarative. The instruction should be precise and representative to inspire the discriminator to think. Extra suggestions are only for reference and can be ignored when they conflict with the overall situation."
    # the instruction of updating examples
    update_dis_ex_instruction_ori = f'''
    Please generate a new example to make LLM discriminator more precise to find LLM generator's errors. The example should be challenging, precise and representative to inspire the discriminator to think. 
    The format of the examples should not be changed.
    Discriminator's instruction is {instruction_discriminator_old}.
    Example must follow this XML format:

    <example>
    <input>[Question]</input>
    <output>[Answer]</output>
    <answer>[Discriminator_Output]</answer>
    </example>

    Replace the content in [] with your output. Provide only the revised `<example>…</example>` blocks.
    '''

    # Obtain modification suggestions through the errors of the discriminator
    update_dis_ins_instruction = update_dis_ins_instruction_ori + get_discriminator_update_suggestion(
        discriminator_update_suggestions_old[:],
        instruction_discriminator_old, 0)

    # Update the instruction of the discriminator
    for i in range(num_iterate_max):
        messages = [{"role": "system", "content": update_dis_ins_instruction},
                    {"role": "user", "content": f"old_instruction:{instruction_discriminator_old}\n new_instruction:"}]

        instruction_discriminator_new = get_llm_answer_content(messages)

        discriminator_new_instruction_prompt = format_messages(instruction_discriminator_new,
                                                               examples_discriminator_old)
        loss_new, generator_update_suggestions_new, discriminator_update_suggestions_new = compute_loss(
            generator_prompt[:], discriminator_new_instruction_prompt[:], true_samples[:],
            train_samples[:])
        print(f"\nUpdate discriminator instruction {i + 1}:\n")
        print(f"new instruction : {instruction_discriminator_new}\n")
        print(f"new loss:{loss_new}\n")
        loss_arr.append(loss_new)
        if loss_new > loss_old:
            loss_old = loss_new
            instruction_discriminator_old = instruction_discriminator_new
            generator_update_suggestions_old = generator_update_suggestions_new
            discriminator_update_suggestions_old = discriminator_update_suggestions_new
            break

    # Update the suggestions of the discriminator's errors
    update_dis_ex_instruction = update_dis_ex_instruction_ori + get_discriminator_update_suggestion(
        discriminator_update_suggestions_old[:],
        instruction_discriminator_old, 1)

    # Update the examples of the discriminator
    for i in range(num_iterate_max):
        examples_discriminator_new = []
        for example_discriminator_single in examples_discriminator_old:
            messages = [{"role": "system", "content": update_dis_ex_instruction},
                        {"role": "user",
                         "content": f"old_example:\nold_example_input:{example_discriminator_single["input"]}\nold_example_output:{example_discriminator_single["output"]}\n new_example:"}]

            res = get_llm_answer_content(messages)
            example_new = replace_discriminator_example(res)

            if example_new["input"] == "" or example_new["output"] == "":
                # If the new example is incorrect, it remains unchanged
                examples_discriminator_new.append(
                    {"input": example_discriminator_single["input"], "output": example_discriminator_single["output"]})
            else:
                examples_discriminator_new.append(example_new)

        discriminator_new_examples_prompt = format_messages(instruction_discriminator_old, examples_discriminator_new)
        loss_new, generator_update_suggestions_new, discriminator_update_suggestions_new = compute_loss(
            generator_prompt[:], discriminator_new_examples_prompt[:], true_samples[:],
            train_samples[:])
        print(f'''\nUpdate discriminator examples {i + 1}:\n''')
        print(f'''new examples: \n{"\n".join(str(example) for example in examples_discriminator_new)}\n''')
        print(f'''new loss:{loss_new}\n''')
        loss_arr.append(loss_new)
        if loss_new > loss_old:
            loss_old = loss_new
            examples_discriminator_old = examples_discriminator_new
            generator_update_suggestions_old = generator_update_suggestions_new
            discriminator_update_suggestions_old = discriminator_update_suggestions_new
            break

    return instruction_discriminator_old, examples_discriminator_old, loss_old, generator_update_suggestions_old, discriminator_update_suggestions_old


def update_generator(instruction_generator_old,
                     examples_generator_old, discriminator_prompt, true_samples, train_samples, loss_old,
                     generator_update_suggestions_old,
                     discriminator_update_suggestions_old, loss_arr,GLOBAL_DATA):
    """
    Update the instructions and examples of the generator
    """
    # The number of candidate instruction and examples
    num_iterate_max = GLOBAL_DATA.get("num_iterate_max")
    # the instruction of updating instruction
    update_gen_ins_instruction_ori = "Try to generate a new generator instruction to improve the correctness of the generator's output. Keep the task instruction as declarative. The instruction should be precise and representative to inspire the generator to think. Extra suggestions are only for reference and can be ignored when they conflict with the overall situation."
    # the instruction of updating examples
    update_gen_ex_instruction_ori = f'''
    Please generate a new example that polish the following example to improve the correctness of the generator's output. The example should be challenging, precise and representative to inspire the generator to think.
    The format of the examples should not be changed.
    Generator's instruction is {instruction_generator_old}.
    Example must follow this XML format:

    <example>
    <input>[Question]</input>
    <output>[Answer]</output>
    </example>

    Replace the content in [] with your output. Provide only the revised `<example>…</example>` blocks.
    '''

    # Obtain modification suggestions through the errors of the generator
    update_gen_ins_instruction = update_gen_ins_instruction_ori + get_generator_update_suggestion(
        generator_update_suggestions_old[:],
        instruction_generator_old, 0)

    # Update the instructions of the generator
    for i in range(num_iterate_max):
        messages = [{"role": "system", "content": update_gen_ins_instruction},
                    {"role": "user", "content": f"old_instruction:{instruction_generator_old}\n new_instruction:"}]
        instruction_generator_new = get_llm_answer_content(messages)
        generator_new_instruction_prompt = format_messages(instruction_generator_new,
                                                           examples_generator_old)
        loss_new, generator_update_suggestions_new, discriminator_update_suggestions_new = compute_loss(
            generator_new_instruction_prompt[:], discriminator_prompt[:], true_samples[:],
            train_samples[:])
        print(f"\nUpdate generator instruction {i + 1}:\n")
        print(f"new instruction : {instruction_generator_new}\n")
        print(f"new loss:{loss_new}\n")
        loss_arr.append(loss_new)
        if loss_new < loss_old:
            loss_old = loss_new
            instruction_generator_old = instruction_generator_new
            generator_update_suggestions_old = generator_update_suggestions_new
            discriminator_update_suggestions_old = discriminator_update_suggestions_new
            break

    # Obtain modification suggestions through the errors of the generator
    update_gen_ex_instruction = update_gen_ex_instruction_ori + get_generator_update_suggestion(
        generator_update_suggestions_old[:],
        instruction_generator_old, 1)

    # Update the examples of the discriminator
    for i in range(num_iterate_max):

        examples_generator_new = []
        for example_generator_single in examples_generator_old:
            messages = [{"role": "system", "content": update_gen_ex_instruction},
                        {"role": "user",
                         "content": f"old_example:\nold_example_input:{example_generator_single["input"]}\nold_example_output:{example_generator_single["output"]}\n new_example:"}]
            example_new = replace_generator_example(get_llm_answer_content(messages))

            if example_new["input"] == "" or example_new["output"] == "":
                # If the new example is incorrect, it remains unchanged
                examples_generator_new.append(
                    {"input": example_generator_single["input"], "output": example_generator_single["output"]})
            else:
                examples_generator_new.append(example_new)

        generator_new_examples_prompt = format_messages(instruction_generator_old, examples_generator_new)
        loss_new, generator_update_suggestions_new, discriminator_update_suggestions_new = compute_loss(
            generator_new_examples_prompt[:], discriminator_prompt[:], true_samples[:],
            train_samples[:])
        print(f"\nUpdate generator examples {i + 1}:\n")
        print(f"new examples: \n{"\n".join(str(example) for example in examples_generator_new)}\n")
        print(f"new loss:{loss_new}\n")
        loss_arr.append(loss_new)
        if loss_new < loss_old:
            loss_old = loss_new
            examples_generator_old = examples_generator_new
            generator_update_suggestions_old = generator_update_suggestions_new
            discriminator_update_suggestions_old = discriminator_update_suggestions_new
            break

    return instruction_generator_old, examples_generator_old, loss_old, generator_update_suggestions_old, discriminator_update_suggestions_old


def format_messages(instruction, examples):
    """
    format messages
    """
    messages = [{"role": "system", "content": instruction}]
    for example in examples:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})
    return messages


def format_messages_optimize_dis(instruction, instruction_old, examples):
    """
    Construct the messages of the optimized discriminator
    """
    messages = [{"role": "system", "content": instruction}]
    query = f"Instruction: {instruction_old}"
    for index, example in enumerate(examples):
        query += f'''Example{index + 1}: {example["input"]} {example["output"]}'''
    messages.append({"role": "user", "content": query})
    return messages


def format_messages_optimize_gen(instruction, instruction_old, examples):
    """
    Construct the messages of the optimized generator
    """
    messages = [{"role": "system", "content": instruction}]
    query = f"Instruction: {instruction_old}"
    for index, example in enumerate(examples):
        query += f'''Example{index + 1}: Input: {example["input"]} Output: {example["output"]}'''
    messages.append({"role": "user", "content": query})
    return messages


def replace_discriminator_example(examples):
    """
    format examples of LLM output when update discriminator
    """

    # Prohibited word
    pattern = r'(?:Question|Generator|Discriminator)'
    if re.search(pattern, examples, flags=re.IGNORECASE):
        return {"input": "", "output": ""}

    text = re.sub(r'\s*\n\s*', ' ', examples)
    m = re.search(r'<example>(.*?)</example>', text, re.IGNORECASE | re.DOTALL)
    if not m:
        return {"input": "", "output": ""}
    block = m.group(1)

    def extract(tag):
        m = re.search(fr'<{tag}>(.*?)</{tag}>', block, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    single_input = extract("input")
    single_output = extract("output")
    single_answer = extract("answer")

    return {
        "input": f"{single_input} Output: {single_output}",
        "output": f"{single_answer}"
    }


def replace_generator_example(examples):
    """
    format examples of LLM output when update generator
    """

    # Prohibited word
    pattern = r'(?:Question|Generator|Discriminator)'
    if re.search(pattern, examples, flags=re.IGNORECASE):
        return {"input": "", "output": ""}

    text = re.sub(r'\s*\n\s*', ' ', examples)
    m = re.search(r'<example>(.*?)</example>', text, re.IGNORECASE | re.DOTALL)
    if not m:
        return {"input": "", "output": ""}
    block = m.group(1)

    inp = re.search(r'<input>(.*?)</input>', block, re.IGNORECASE | re.DOTALL)
    outp = re.search(r'<output>(.*?)</output>', block, re.IGNORECASE | re.DOTALL)
    return {
        "input": inp.group(1).strip() if inp else "",
        "output": outp.group(1).strip() if outp else ""
    }