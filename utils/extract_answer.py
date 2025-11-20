import re

from utils.extract_number import extract_number, extract_letter
from utils.llm_answer import get_llm_answer_content


def evaluate_answer(generator_output, sample_input, dataset_type, sample_output):
    """
    extract answer through LLM
    """
    if generator_output == "":
        return False

    # method for gsm8k/svamp/multiArith
    if "gsm" in dataset_type or "svamp" in dataset_type or "multiArith" in dataset_type:
        generator_output = generator_output.replace("\n", " ")
        instruction_extractor = "There is a pair of input and output. Please extract the final answer from the output. The answer should be a number, which may be negative, positive, decimal or zero. Characters and punctuation marks should not appear in the answer. If the answer doesn't exist, output Err. If all the digits after the decimal point are 0, only output the integer part. Please output the number directly. Answer:"
        input_temp = f"input:{sample_input} output:{generator_output}"
        messages = [{"role": "system", "content": instruction_extractor}, {"role": "user", "content": input_temp}]

        final_answer = get_llm_answer_content(messages, 0)
        final_answer = extract_number(final_answer)
        print(f"final_answer:{final_answer}")
        print(f"sample_output:{sample_output}")
        if str(final_answer) == str(sample_output):
            return True
        else:
            return False

    # method for choice
    elif "aqua" in dataset_type or "arc" in dataset_type or "OpenBook" in dataset_type or "CSQA" in dataset_type:
        patterns = [
            r'answer is:.*\(?([A-E])\)?',
            r'answer is:?\s+\(?([A-E])\)?',
            r'answer:?\s+\(?([A-E])\)?',
            r'be:?\s+\(([A-E])\)',
            r'is:?\s+\(([A-E])\)',
            r'is:?.*\(([A-E])\)',
            r':\s+\(?([A-E])\)?',
            r'\(([A-E])\)',
        ]

        final_answer = ""

        for pattern in patterns:
            match = re.search(pattern, generator_output)
            if match:
                final_answer = match.group(1)
                break

        # final_answer = extract_letter(final_answer)

        print(f"final_answer:{final_answer}")
        print(f"sample_output:{sample_output}")
        if str(final_answer).lower() == str(sample_output).lower():
            return True
        else:
            return False

    # method for yes_or_no
    elif "coin" in dataset_type or "boolq" in dataset_type or "sports" in dataset_type or "Strategy" in dataset_type:
        patterns = [
            r'\s+answer\s+is\s*(?::\s*|\(\s*|\s+)?(yes|no)\b\s*[):]?',
            r'answer.?\s+[\'"]?(yes|no)[\'"]?',
            r'choice.?\s+[\'"]?(yes|no)[\'"]?',
            r'option.?\s+[\'"]?(yes|no)[\'"]?',
            r':\s+[\'"]?(yes|no)[\'"]?',
            r'\b(yes|no)\b'
        ]

        final_answer = ""

        for pattern in patterns:
            match = re.search(pattern, generator_output, re.IGNORECASE)
            if match:
                final_answer = match.group(1)
                break

        print(f"final_answer:{final_answer}")
        print(f"sample_output:{sample_output}")
        if str(final_answer).lower() == str(sample_output).lower():
            return True
        else:
            return False

    # method for letter
    elif "letter" in dataset_type:
        patterns = [
            r'\s+answer\s+is\s*(?::\s*|\(\s*|\s+)?([a-zA-Z]+)\b\s*[):]?',
            r'answer.?\s+[\'"]?([a-zA-Z]+)[\'"]?',
            r'choice.?\s+[\'"]?([a-zA-Z]+)[\'"]?',
            r'option.?\s+[\'"]?([a-zA-Z]+)[\'"]?',
            r':\s+[\'"]?([a-zA-Z]+)[\'"]?',
            r'\b([a-zA-Z]+)\b'
        ]
        final_answer = ""
        for pattern in patterns:
            match = re.search(pattern, generator_output, re.IGNORECASE)
            if match:
                final_answer = match.group(1)
                break
        print(f"final_answer:{final_answer}")
        print(f"sample_output:{sample_output}")
        if str(final_answer).lower() == str(sample_output).lower():
            return True
        else:
            return False

    else:
        return False