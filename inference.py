import os
import sys
import time
import pandas as pd

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import gc

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

input_prompt =  '''You are an AI assistant designed to help developers in their day to day tasks by automating their requests using tools available to them.

The actions defined in this domain are:

The works_list action: Returns a list of work items matching the request. It has the following parameters: 
	applies_to_part: Filters for work belonging to any of the provided parts.
	created_by: Filters for work created by any of these users.
	issue.priority: Filters for issues with any of the provided priorities. Allowed values: p0, p1, p2, p3.
	issue.rev_orgs: Filters for issues with any of the provided Rev organizations.
	limit: The maximum number of works to return. The default is '50'.
	owned_by: Filters for work owned by any of these users.
	stage.name: Filters for records in the provided stage(s) by name.
	ticket.needs_response: Filters for tickets that need a response.
	ticket.rev_org: Filters for tickets associated with any of the provided Rev organizations.
	ticket.severity: Filters for tickets with any of the provided severities. Allowed values: blocker, high, low, medium.
	ticket.source_channel: Filters for tickets with any of the provided source channels.
	type: Filters for work of the provided types. Allowed values: issue, ticket, task.


The summarize_objects action: Summarizes a list of objects. The logic of how to summarize a particular object type is an internal implementation detail. It has the following parameters: 
	objects: List of objects to summarize.


The prioritize_objects action: Returns a list of objects sorted by priority. The logic of what constitutes priority for a given object is an internal implementation detail. It has the following parameters: 
	objects: A list of objects to be prioritized.


The add_work_items_to_sprint action: Adds the given work items to the sprint. It has the following parameters: 
	work_ids: A list of work item IDs to be added to the sprint.
	sprint_id: The ID of the sprint to which the work items should be added.


The get_sprint_id action: Returns the ID of the current sprint. It has no parameters. 

The get_similar_work_items action: Returns a list of work items that are similar to the given work item. It has the following parameters: 
	work_id: The ID of the work item for which you want to find similar items.


The search_object_by_name action: Given a search string, returns the ID of a matching object in the system of record. If multiple matches are found, it returns the one where the confidence is highest. It has the following parameters: 
	query: The search string, could be for example customer’s name, part name, user name.


The create_actionable_tasks_from_text action: Given a text, extracts actionable insights, and creates tasks for them, which are kind of a work item. It has the following parameters: 
	text: The text from which the actionable insights need to be created.


The who_am_i action: Returns the ID of the current user. It has no parameters. 
'''


def main(
    load_8bit: bool = False,
    base_model: str = "llama2-platypus-7B",
    lora_weights: str = "",
    prompt_template: str = "alpaca",
    csv_path: str = "test.csv",
    output_csv_path: str = "test-results.csv"
):
    
    base_model = base_model or os.environ.get("BASE_MODEL", "")

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # df = pd.read_csv(csv_path)
    df = pd.DataFrame([
    {
        'instruction': input("Enter query: "),
        'input': input_prompt
    }
])

    instructions = df["instruction"].tolist()
    inputs = df["input"].tolist()

    results = []
    max_batch_size = 16
    for i in range(0, len(instructions), max_batch_size):
        instruction_batch = instructions[i:i + max_batch_size]
        input_batch = inputs[i:i + max_batch_size]
        print(f"Processing batch {i // max_batch_size + 1} of {len(instructions) // max_batch_size + 1}...")
        start_time = time.time()
    
        prompts = [prompter.generate_prompt(instruction, None) for instruction, input in zip(instruction_batch, input_batch)]
        batch_results = evaluate(prompter, prompts, model, tokenizer)
            
        results.extend(batch_results)
        print(f"Finished processing batch {i // max_batch_size + 1}. Time taken: {time.time() - start_time:.2f} seconds")

    print(results)

def evaluate(prompter, prompts, model, tokenizer):
    batch_outputs = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        generation_output = model.generate(input_ids=input_ids, num_beams=1, num_return_sequences=1,
                                           max_new_tokens=2048, temperature=0.15, top_p=0.95)
        
        output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        resp = prompter.get_response(output)
        print(resp)
        batch_outputs.append(resp)

    return batch_outputs


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)

