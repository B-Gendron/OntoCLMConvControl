import torch
from tqdm import tqdm
from termcolor import colored
import json
import os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, matthews_corrcoef, mean_absolute_error, mean_squared_error

from utils import *


def format_generation(generation):
    """
    Formats the generated response of the agent by ensuring the last sentence is complete. Removes unfinished sentences to make the agent's response more realistic and adds a prefix indicating the agent.

    Args:
        generation (str): The raw generated text.

    Returns:
        str: The formatted response with a complete sentence and prefixed by 'Agent >'.
    """
    last_period_index = generation.rfind('.')
    
    if last_period_index != -1:
        generation = generation[:last_period_index + 1].strip()
    
    return f'Agent > {generation}'

def evaluate_one_iter(args, experiment, it, ontology_concept, finetuning=True):
    """
    Evaluates one iteration of the generation process controlled by ontology-based labels.
    Saves the generated sentences, creates batch individuals, and retrieves predicted labels.

    Args:
        args (Namespace): Arguments required for generation and evaluation.
        experiment (str): The name of the experiment.
        it (int): The current iteration index.
        ontology_concept (str): The concept being evaluated, e.g., 'ProficiencyLevel' or 'NextIntention'.
        finetuning (bool, optional): Whether fine-tuning is applied. Defaults to True.

    Returns:
        tuple: A pair containing:
            - asked_labels (list): The actual labels of the sentences.
            - got_labels (list): The labels predicted based on the ontology.
    """

    # generate sentences with a specific RL
    asked_labels, batch_generations = generate_from_random_prompts(args, ontology_concept, finetuning)
    file_path = save_batch_generations(asked_labels, batch_generations, it, ontology_concept, experiment)

    # trues are actual labels of the sentences
    create_batch_individual(it, file_path, experiment, "fine-tuning", ontology_concept)
    if ontology_concept == "ProficiencyLevel":
        got_labels = get_proficiency_levels(f'../rdf/individual_batch_{it}_{experiment}.rdf', ontology_concept)
    
    elif ontology_concept == "NextIntention":
        got_labels = get_emotion_labels(f'../rdf/individual_batch_{it}_{experiment}.rdf', ontology_concept)

    else:
        print("Unknown ontology concept. Valid ontology concepts are 'ProficiencyLevel' and 'NextIntention'.")

    return asked_labels, got_labels


def evaluate_one_iter_conv(args, experiment, it, ontology_concept):
    """
    An evaluation step in a conversational setting. Handles user interaction and displays the required class for the agent to generate, according to the user content.

    Steps:
        1. Prompts the user for input.
        2. Determines the CEFR level or intention using the ontology.
        3. Selects the appropriate prefix based on the level or intention.
        4. Generates and formats a response.
        5. Updates the conversation history.

    Args:
        args (Namespace): Arguments required for generation and evaluation.
        experiment (str): The name of the experiment.
        it (int): The current iteration index. Resets history if `it == 0`.
        ontology_concept (str): The concept being evaluated, that can be 'ProficiencyLevel' or 'NextIntention'.
    """
    proficiency_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
    emotion_mapping = {0: 'LoadedPositiveIntention', 1: 'LoadedNegativeIntention', 2: 'NonLoadedPositiveIntention', 3: 'NonLoadedNegativeIntention'}
    file_path = f'../objects/conversations/conv.txt'
    history_file_path = f'../objects/conversations/{experiment}.txt'
    # if it is a new conversation, clear the file that store user prompts
    if it == 0:
        open(file_path, 'w').close() 

    # get the user prompt
    user_prompt = input("User > ")
    with open(file_path, 'a') as f:
        f.write(user_prompt + '\n')

    create_batch_individual(it, f'../fine-tuning/{file_path[2:]}', experiment, "fine-tuning", ontology_concept)

    if ontology_concept == "ProficiencyLevel":
        next_cefr_index = get_next_proficiency_level(f'../rdf/individual_batch_{it}_{experiment}.rdf')
        print(colored(f"Next CEFR level needs to be {proficiency_mapping[next_cefr_index]}", 'red'))
        _, generation = generate_from_next_cefr_index(args, next_cefr_index, user_prompt)
        print(format_generation(generation))

    elif ontology_concept == "NextIntention":
        next_intention_index = get_next_intention(f'../rdf/individual_batch_{it}_{experiment}.rdf')
        print(colored(f"Next Intention should be {emotion_mapping[next_intention_index]}", 'red'))
        _, generation = generate_from_next_intention(args, next_intention_index, user_prompt, experiment)
        print(format_generation(generation))

    # update conversation history with the new content
    with open(history_file_path, 'a') as f:
        f.write('> user ' + user_prompt + '\n')
        f.write('> agent ' + generation + '\n')


def evaluate(args, experiment, ontology_concept, finetuning=True):
    """
    Evaluates the performance of a model over multiple iterations, computing classification metrics and saving the results to a JSON file.

    Args:
        args (dict): Dictionary of evaluation arguments, including 'eval_iters' for the number of iterations.
        experiment (str): The name of the experiment.
        ontology_concept (str): The ontology concept being evaluated, e.g., 'ProficiencyLevel' or 'NextIntention'.
        finetuning (bool, optional): Whether the model is fine-tuned. Defaults to True.
    """
    n = args['eval_iters']
    all_asked, all_got = [], []

    for it in tqdm(range(n)):
        asked, got = evaluate_one_iter(args, experiment, it, ontology_concept, finetuning)
        all_asked.extend(asked)
        all_got.extend(got)

    # compute classification metrics
    metrics = {
    'acc': accuracy_score(all_asked, all_got),
    'prec': precision_score(all_asked, all_got, average='weighted'),
    'rec': recall_score(all_asked, all_got, average='weighted'),
    'f1':f1_score(all_asked, all_got, average='weighted'),
    'mcc':matthews_corrcoef(all_asked, all_got),
    'mae':mean_absolute_error(all_asked, all_got),
    'mse':mean_squared_error(all_asked, all_got)
    }

    # dump into json
    with open(f'../results/experiments/{experiment}.json', 'w') as f:
        json.dump(metrics, f)

    # return displayable versions of the metrics
    print(classification_report(all_asked, all_got), confusion_matrix(all_asked, all_got))


def evaluate_conv(args, experiment, ontology_concept):
    """
    Conducts an interactive evaluation of a model in a conversational setting.

    First, it iteratively calls `evaluate_one_iter_conv` for `n` iterations, where `n` is specified in the evaluation arguments. Then, it simulates a conversational scenario for each iteration, interacting with the user and generating responses based on the ontology concept.

    Args:
        args (dict): Dictionary of evaluation arguments, including 'eval_iters' for the number of iterations.
        experiment (str): The name of the experiment.
        ontology_concept (str): The ontology concept being evaluated, e.g., 'ProficiencyLevel' or 'NextIntention'.
    """
    n = args['eval_iters']

    for it in range(n):
        evaluate_one_iter_conv(args, experiment, it, ontology_concept)


if __name__ == "__main__":

    args = {
    'device': activate_gpu(),
    'eval_iters': 100,
    'batch_size': 32,
    'max_new_tokens':100,
}

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct" # select the LLM to evaluate
    adapter_model_name = "../adapters/llama3_cefr/checkpoint-165500"  # put the right checkpoint here

    clm_model = AutoModelForCausalLM.from_pretrained(  
        model_name,
        low_cpu_mem_usage=True,         # recommanded param
        return_dict=True,               # not used for now
        torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
        device_map=args['device'],      # send to the right device
    )

    clm_model = PeftModel.from_pretrained(clm_model, adapter_model_name)

    args.update({
        'model': clm_model,
        'tokenizer': AutoTokenizer.from_pretrained(model_name, padding_side='left'),
    })

    # set this parameter to True to use the models as conversational agents
    conversation = False

    if conversation:
        args.update({
            'eval_iters': 5, # very small number to evaluate on a conversation
            'max_new_tokens': 100 # a longer output is possible in this scenario
        })

    args['tokenizer'].pad_token_id = args['tokenizer'].eos_token_id

    if conversation:
        evaluate_conv(args, f'emotion_conv_test', ontology_concept="NextIntention")
    else:
        evaluate(args, f'cefr_llama3', ontology_concept='ProficiencyLevel')