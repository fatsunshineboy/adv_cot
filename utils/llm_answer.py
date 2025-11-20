import configparser
import os
import time
from configparser import NoOptionError

from openai import APIError, AuthenticationError
from openai import OpenAI
from openai.types.chat import ChatCompletion


def get_llm_answer_content(messages, temperature=0.6, seed_flag=False, max_retries=3):
    llm_answer = get_llm_answer(messages, temperature=temperature, seed_flag=seed_flag, max_retries=max_retries)
    # print(llm_answer)
    if isinstance(llm_answer, ChatCompletion) and llm_answer.choices[0].message.content is not None:
        res = llm_answer.choices[0].message.content.replace('\n\n', '\n').replace('\n', ' ')
    else:
        res = ""
    return res


def get_llm_answer(messages, temperature=0.6, seed_flag=False, max_retries=3):
    # print(f'{"-" * 30} User question {"-" * 30}\n{messages}\n')

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    CONFIG_DIR = os.path.join(ROOT_DIR, "config")
    CONFIG_PATH = os.path.join(CONFIG_DIR, "config.ini")

    # load config
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    # base_url
    base_url = config.get('OpenAI', 'base_url')
    # model
    model_name = config.get('OpenAI', 'model')
    # api_key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        api_key = config.get('OpenAI', 'api_key')

    if not api_key:
        print("No valid API key was found")
        return ""

    client = None
    retry_count = 0
    last_exception = None

    while retry_count < max_retries:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)

            completion_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "logprobs": True,
                "top_logprobs": 2
            }

            if seed_flag:
                completion_params["seed"] = 42

            completion = client.chat.completions.create(**completion_params)

            if isinstance(completion, ChatCompletion):
                # print(f"{completion.choices[0].message.content}")
                # print(f'{"-" * 30} End Answer {"-" * 30}\n')
                return completion

        except (APIError, AuthenticationError) as e:
            last_exception = e
            # print(f"API call failed. (Key:{api_key[:8]}...):{str(e)}")
            # time.sleep(1)
            continue
        except Exception as e:
            last_exception = e
            print(f"Unknown error: {str(e)}")
            time.sleep(1)
            continue

        retry_count += 1
        print(f"Retry count: {retry_count}/{max_retries}")
        time.sleep(2)

    print("All retry attempts have been exhausted. Final error:", str(last_exception))
    return ""