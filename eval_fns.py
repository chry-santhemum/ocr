import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
ds_path = "/workspace/inductive-oocr/functions/dev/047_functions/finetune_01"

def load_functions_testset(path):
    # each row: {"messages": [message dicts]}
    # this doesn't need any additional preprocessing with SFTTrainer
    ds = []

    output = []
    ans = []
    with open(path, 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    # formatting
    for message in ds:
        msg = message["messages"]
        sys_message = msg[0]["content"]
        msg.pop(0)
        msg[0]["content"] = sys_message + "\n" + msg[0]["content"] + "\n" + msg[1]["content"]

        ans.append(msg[-1]["content"])
        msg.pop(-1)
        msg.pop(-1)
        output.append(msg)

    return output, ans


def extract_answer(text):
    start_tag = "<start_of_turn>model"
    
    start_index = text.find(start_tag)
    if start_index == -1:
        return None
    
    # Move past the start tag
    start_index += len(start_tag)
    
    # Look for the first capital letter A-E after the start tag
    for i in range(start_index, len(text)):
        if text[i] in "ABCDE":
            return text[i]
    
    # No capital letter A-E found
    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()
    
    test_dataset, ans = load_functions_testset(os.path.join(ds_path, "047_func_01_test_oai.jsonl"))

    # load the checkpoint
    # "checkpoints/2b-functions-full/checkpoint-4000/"

    model_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True,
    )
    model.eval()

    input_ids = tokenizer.apply_chat_template(test_dataset, tokenize=True, add_generation_prompt=True, return_tensors='pt', padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids.to("cuda"),
            max_length=150, 
            do_sample=False,
        )

    outputs = outputs[:, 128:]
    decoded_outputs = [tokenizer.decode(outputs[i,:]) for i in range(outputs.shape[0])]
    total = len(decoded_outputs)

    # # print some random completions
    # print("="*50)
    # for i in range(5):
    #     print(decoded_outputs[i])
    #     print("-"*50)

    model_ans = [extract_answer(decoded_outputs[i]) for i in range(total)]

    correct = [ans[i]==model_ans[i] for i in range(total)]
    score = sum(correct)/total

    print("Number of questions:", total)
    print("Accuracy:", score)