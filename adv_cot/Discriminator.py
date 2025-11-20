import math

from adv_cot.Generator import get_generator_output
from utils.llm_answer import get_llm_answer
from openai.types.chat import ChatCompletion

def compute_loss(generator_prompt, discriminator_prompt, true_samples,
                 train_samples):
    """
    Loss function
    """
    loss_score = 0
    # the collection of generator error
    generator_update_suggestions = []
    # the collection of discriminator error
    discriminator_update_suggestions = []

    # For artificially designed answers, the discriminator should output true
    for sample in true_samples:

        log_probability, verify_flag, answer = get_discriminator_output(discriminator_prompt[:], sample)

        loss_score += log_probability
        # If the discriminator considers it false, the discriminator should be corrected
        if not verify_flag:
            discriminator_update_suggestions.append(
                {"input": sample["input"], "output": sample["output"], "answer": answer})

    # For LLM answers, the discriminator should output false
    for sample in train_samples:
        generator_output = get_generator_output(generator_prompt[:], sample["input"])
        log_probability, verify_flag, answer = get_discriminator_output(discriminator_prompt[:],
                                                                        {"input": sample["input"],
                                                                         "output": generator_output})
        loss_score += math.log(1 - math.exp(log_probability))
        # If the discriminator considers it true, the discriminator should be corrected
        if verify_flag:
            discriminator_update_suggestions.append(
                {"input": sample["input"], "output": generator_output, "answer": answer})
        # If the discriminator considers it false, the generator should be corrected
        else:
            generator_update_suggestions.append(
                {"input": sample["input"], "output": generator_output, "answer": answer})

    loss_score = loss_score / (len(true_samples) + len(train_samples))
    return loss_score, generator_update_suggestions, discriminator_update_suggestions

def get_discriminator_output(discriminator_prompt, sample):
    """
    Return the confidence level of output A of the discriminator
    """
    user_query = f'''input: {sample["input"]} output: {sample["output"]}'''
    addition = 'Is above output correct ground truth?' + "\n(A) Yes, it is correct ground truth.\n(B) No, it is generated fake output.\nThe answer is:"
    user_query += addition

    discriminator_prompt.append({"role": "user", "content": user_query})

    llm_answer = get_llm_answer(discriminator_prompt, 0)

    if isinstance(llm_answer, ChatCompletion) and llm_answer.choices[0].message.content is not None:
        discriminator_output = llm_answer.choices[0].message.content.replace('\n\n', '\n').replace('\n', ' ')
    else:
        return -10, False, ""
    llm_answer = llm_answer.choices[0]

    target_tokens = ['A', 'B']
    found_logprob = None
    token = None
    if llm_answer.logprobs and llm_answer.logprobs.content:
        # Search for tokens in order of priority
        for token in target_tokens:
            for token_logprob in llm_answer.logprobs.content:
                if token_logprob.token.strip().strip("()").strip() == token:
                    found_logprob = token_logprob.logprob
                    break
            if found_logprob is not None:
                break

    # When verify_flag is True, it indicates that the discriminator considers the answer to be true
    verify_flag = False
    # Default confidence level
    log_probability = -10

    if found_logprob is not None and found_logprob < 0:
        if token == "A":
            log_probability = found_logprob
            verify_flag = True
        else:
            # If token is B, then convert it to the confidence level of A
            log_probability = math.log(1 - math.exp(found_logprob))
    else:
        log_probability = -10

    return log_probability, verify_flag, discriminator_output