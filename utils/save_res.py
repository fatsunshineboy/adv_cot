import datetime
import os


def save_result_to_file(topic, custom_message, dir_path):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:

        with open(os.path.join(dir_path, "result.txt"), "a", encoding="UTF-8") as file:
            file.write(f"Time: {current_time}\n")
            file.write(f"Subject: {topic}\n")
            file.write(f"Content: {custom_message}\n")
            file.write("-" * 50 + "\n")
    except Exception as e:
        print(f"An error occurred when saving the file: {e}")

def save_err_to_file(sample_id, input_temp, output, dir_path):
    try:

        with open(os.path.join(dir_path, "err.txt"), "a", encoding="UTF-8") as file:
            file.write(f"Id: {sample_id}\n")
            file.write(f"Input: {input_temp}\n")
            file.write(f"Output: {output}\n")
            file.write("-" * 50 + "\n")
    except Exception as e:
        print(f"An error occurred when saving the file: {e}")