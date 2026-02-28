import os
from owlready2 import *
from termcolor import colored
from datetime import datetime
from datasets import DatasetDict
import torch
import re
import random
import csv

# set default tensor type
torch.set_default_dtype(torch.float16)

cefr_starters = [
    "Write a short story about a dragon and a knight who become friends.",
    "Explain how blockchain technology works.",
    "What were the main causes and effects of the French Revolution?",
    "Describe the process of photosynthesis.",
    "What is the meaning of life according to existential philosophy?",
    "Analyze the themes of '1984' by George Orwell.",
    "Write a poem about the ocean at sunrise.",
    "What are the advantages and disadvantages of nuclear energy?",
    "Summarize the plot of 'Pride and Prejudice'.",
    "Describe the major differences between classical and operant conditioning.",
    "What are the key features of quantum computing?",
    "Explain the significance of the Theory of Relativity.",
    "Write a dialogue between two characters discussing artificial intelligence.",
    "What are the economic impacts of climate change?",
    "Describe the main components of a healthy diet.",
    "Explain the concept of supply and demand in economics.",
    "What are the ethical implications of genetic engineering?",
    "Describe the cultural impact of the Beatles on modern music.",
    "Write a news article about a recent scientific discovery.",
    "What are the primary symptoms and treatments for depression?",
    "Explain the steps involved in the water cycle.",
    "What are the fundamental principles of democracy?",
    "Write a speech advocating for renewable energy sources.",
    "Describe the process of mitosis in cell division.",
    "What are the historical roots of the conflict in the Middle East?",
    "Explain how the internet has changed global communication.",
    "Write a review of a popular movie or TV show.",
    "What are the causes and effects of deforestation?",
    "Describe the role of women in ancient Greek society.",
    "What are the key factors in maintaining mental health?"
]

emo_starters = [
    "How do you feel about rainy days?",
    "Describe your favorite memory.",
    "What’s your opinion on early mornings?",
    "Tell me about a time you felt truly happy.",
    "What comes to mind when you think of failure?",
    "Do you enjoy meeting new people?",
    "How do you handle criticism?",
    "What do you think of surprises?",
    "Share your thoughts on family gatherings.",
    "What do you usually do when you’re alone?",
    "How do you feel about public speaking?",
    "What’s your opinion on taking risks?",
    "Tell me about a time you felt disappointed.",
    "What do you think about people who work too much?",
    "How do you feel when someone doesn’t keep a promise?",
    "What’s your reaction to unexpected compliments?",
    "How do you usually feel on your birthday?",
    "What do you think of long commutes?",
    "Tell me about something you regret.",
    "How do you feel when you accomplish a goal?",
    "What’s your opinion on social media?",
    "What do you think about trying new food?",
    "How do you feel when someone helps you unexpectedly?",
    "Tell me about a time you felt proud of yourself.",
    "What do you think about going to the movies alone?",
    "How do you feel when someone disagrees with you?",
    "What’s your opinion on forgiveness?",
    "How do you react when someone interrupts you?",
    "What comes to mind when you think of home?",
    "How do you feel about changes in your routine?"
]

# -----------------------------------------------------------------------------------------
# General purpose auxiliary functions
# -----------------------------------------------------------------------------------------

def get_datetime():
    '''
    Get the current date and time and return it as a string.

    Returns:
        str: The current time formatted as a string.
    '''
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    return dt_string


def activate_gpu(force_cpu=False):
    '''
    Return the appropriate device based on GPU availability.

    Args:
        force_cpu (bool): A flag to force using the CPU, regardless of GPU availability.

    Returns:
        str: The name of the selected device ('cuda', 'mps', or 'cpu').
    '''
    device = "cpu"
    if not force_cpu:
        if torch.cuda.is_available():
            device = 'cuda'
            print('DEVICE = ', colored(torch.cuda.get_device_name(0), "green" ) )
        elif torch.backends.mps.is_available():
            device = 'mps'
            print('DEVICE = ', colored("mps", "green" ) )
        else:
            device = 'cpu'
            print('DEVICE = ', colored('CPU', "blue"))
    return device


def remove_bracketed_text(sentence):
    '''
    Remove bracketed text from a sentence.

    Args:
        sentence (str): The input sentence from which to remove bracketed text.

    Returns:
        str: The sentence without bracketed text.
    '''
    return re.sub(r'\[.*?\]', '', sentence)

def load_dataset(dataset_name):
    '''
    Load a dataset saved in the HuggingFace format, located in the data folder.

    Args:
        dataset_name (str): The name of the dataset, corresponding to an existing folder inside the data folder.

    Returns:
        DatasetDict: The dataset as a DatasetDict object.

    Raises:
        Exception: If no folder with the specified name is found in the data folder.
    '''
    path = f"../data/{dataset_name}"
    if os.path.isdir(path):
        dataset = DatasetDict.load_from_disk(f"../data/{dataset_name}")
    else: 
        raise Exception("No folder with the such name found in the data folder.")

    return dataset

def save_dataset(dataset, dataset_name, output_format='huggingface'):
    '''
    Save the dataset into a HuggingFace format using the `save_to_disk` method.

    Args:
        dataset (DatasetDict): The dataset to save in a DatasetDict format.
        dataset_name (str): The name to give to the file.
        output_format (str): Either 'huggingface' or 'json'. Default is 'huggingface'.

    Prints:
        A message indicating the accepted formats if an unrecognized format is given.
    '''
    if output_format == "huggingface":
        dataset.save_to_disk(f'../data/{dataset_name}')
    elif output_format == "json":
        with open(f"../data/{dataset_name}.json", 'w') as f:
            json.dump(dataset, f)
    else:
        print("The given output format is not recognized. Please note that accepted formats are 'huggingface' and 'json")


def save_batch_generations(batch_labels, batch_generations, batch_index, ontology_concept, experiment):
    '''
    Save batch generations to a .tsv file.

    Args:
        batch_labels (list): Labels corresponding to the batch generations.
        batch_generations (list): Generated text for each label.
        batch_index (int): Index of the current batch.
        ontology_concept (str): Concept type (e.g., "ProficiencyLevel", "NextIntention").
        experiment (str): Experiment identifier.

    Returns:
        str: Relative file path to the saved TSV file.
    '''
    if ontology_concept == "ProficiencyLevel":
        label_mapping = {0:'A1', 1:'A2', 2:'B1', 3:'B2', 4:'C1', 5:'C2'}
    elif ontology_concept == "NextIntention":
        label_mapping = {0:'LoadedPositive', 1:'LoadedNegative', 2:'LoadedNeutral', 3:'NonLoadedPositive', 4:'NonLoadedNegative', 5:'NonLoadedNeutral'}
    file_path = f'../objects/batch_generations_{batch_index}_{experiment}.tsv'
    with open(file_path, 'w', newline='') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        for i in range(len(batch_generations)):
            row, label = batch_generations[i], batch_labels[i]
            row = row.replace('\n', '\\n')
            output = f'Asked: {label_mapping[label]} | {row}'
            tsv_writer.writerow([output])

    return file_path[3:] # remove ../ to make it relative


# -----------------------------------------------------------------------------------------
# Preprocessing utils for training
# -----------------------------------------------------------------------------------------

def parse_indexes(levels_dict, ontology_concept):
    '''
    Parse and retrieve utterance indexes from the readability level dictionary.

    Args:
        levels_dict (dict): Dictionary with ontology identifiers.
        ontology_concept (str): Concept type (e.g., "NextIntention").

    Returns:
        dict: Dictionary mapping indexes to readability levels.
    '''
    levels_indexes_dict = {}

    # iterate through the input dict
    for k in levels_dict.keys():
        for v in levels_dict[k]:
            # parse the utterance full name to retrieve its index
            splitted_v = re.split(r'[_\.]', v)[::-1]
            index = splitted_v[1] if ontology_concept == 'NextIntention' else splitted_v[2] # the 3rd element when individual_592.592_598_utt_21271711 is splitted by _ is 598, which is the utterance index
            # store the readability class for the current index
            levels_indexes_dict[int(index)] = k

    result_dict = dict(sorted(levels_indexes_dict.items()))

    return result_dict


def get_proficiency_levels(indiv_path, ontology_concept):
    '''
    Retrieves and maps the proficiency levels for a given ontology individual.
    
    Parameters:
        indiv_path (str): Path to the ontology individual.
    
    Returns:
        labels (list): A list of CEFR proficiency labels (A1, A2, B1, B2, C1 or C2)
        rough_labels (list): A list of simplified proficiency labels (A, B or C)
    '''
    # get labels mapping for different readability level
    proficiency_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()
    # crop first element as it simply corresponds to the class occurence
    utterance_levels = {
        'A1':       [str(i) for i in individual.search(is_a=individual.A1)[1:]],
        'A2':       [str(i) for i in individual.search(is_a=individual.A2)[1:]],
        'B1':       [str(i) for i in individual.search(is_a=individual.B1)[1:]],
        'B2':       [str(i) for i in individual.search(is_a=individual.B2)[1:]],
        'C1':       [str(i) for i in individual.search(is_a=individual.C1)[1:]],
        'C2':       [str(i) for i in individual.search(is_a=individual.C2)[1:]]
    }
    utterance_levels = parse_indexes(utterance_levels, ontology_concept)
    labels = [proficiency_mapping[v] for v in utterance_levels.values()]

    individual.destroy()

    return labels


def get_emotion_labels(indiv_path, ontology_concept):
    '''
    Retrieves and maps the proficiency levels for a given ontology individual.
    
    Parameters:
        indiv_path (str): Path to the ontology individual.
    
    Returns:
        labels (list): A list of CEFR proficiency labels (A1, A2, B1, B2, C1 or C2)
        rough_labels (list): A list of simplified proficiency labels (A, B or C)
    '''
    # get labels mapping for different readability level
    proficiency_mapping = {'LoadedPositive': 0, 'LoadedNegative': 1, 'LoadedNeutral': 2, 'NonLoadedPositive': 3, 'NonLoadedNegative': 4, 'NonloadedNeutral': 5}
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()

    utterance_levels = {
        'LoadedPositive':       [str(i) for i in individual.search(is_a=individual.LoadedPositiveUtterance)[1:]],
        'LoadedNegative':       [str(i) for i in individual.search(is_a=individual.LoadedNegativeUtterance)[1:]],
        'LoadedNeutral':        [str(i) for i in individual.search(is_a=individual.LoadedNeutralUtterance)[1:]],
        'NonLoadedPositive':    [str(i) for i in individual.search(is_a=individual.NonLoadedPositiveUtterance)[1:]],
        'NonLoadedNegative':    [str(i) for i in individual.search(is_a=individual.NonLoadedNegativeUtterance)[1:]],
        'NonloadedNeutral':     [str(i) for i in individual.search(is_a=individual.NonLoadedNeutralUtterance)[1:]],
    }
    utterance_levels = parse_indexes(utterance_levels, ontology_concept)
    labels = [proficiency_mapping[v] for v in utterance_levels.values()]

    individual.destroy()

    return labels


def format_user_history(user_prompt, experiment):
    '''
    Format user history with or without existing conversation history.

    Args:
        user_prompt (str): The current user prompt.
        experiment (str): Experiment identifier.

    Returns:
        str: Formatted conversation history with the user prompt.
    '''
    path_to_conv_history = f'../objects/conversations/{experiment}.txt'
    
    # check if there is already a conversation history
    if os.path.exists(path_to_conv_history):
        with open(path_to_conv_history, 'r') as f:
            conversation_history = f.read().strip()
    else:
        conversation_history = ""
    
    # format the output for the prompt with or without history
    if conversation_history:
        formatted_history = f"{conversation_history}\n{user_prompt}"
    else: 
        formatted_history = f"{user_prompt}"
    
    return formatted_history


def get_next_proficiency_level(indiv_path):
    '''
    Retrieves and maps the next proficiency level to be used in the conversation for a given ontology individual. Here we consider that there is only one conversation
    
    Parameters:
        indiv_path (str): Path to the ontology individual.
    
    Returns:
        labels (list): A list of CEFR proficiency labels (A1, A2, B1, B2, C1 or C2)
        rough_labels (list): A list of simplified proficiency labels (A, B or C)
    '''
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()

    # crop first element as it simply corresponds to the class occurence
    conversation = individual.search(is_a=individual.Conversation)[1:][0]

    # get the classes the conversation belongs to and retrieve the class of interest (CEFR level)
    conv_types = [str(typ)[str(typ).find('.')+1:] for typ in conversation.is_a]
    possible_inferred_types = ['A1LevelConversation', 'A2LevelConversation', 'B1LevelConversation', 'B2LevelConversation', 'C1LevelConversation', 'C2LevelConversation']
    for i in range(len(possible_inferred_types)):
        if possible_inferred_types[i] in conv_types:
            label = i
        
    individual.destroy()

    return label


def get_next_intention(indiv_path):
    '''
    Retrieves and maps the next proficiency level to be used in the conversation for a given ontology individual. Here we consider that there is only one conversation
    
    Parameters:
        indiv_path (str): Path to the ontology individual.
    
    Returns:
        labels (list): A list of CEFR proficiency labels (A1, A2, B1, B2, C1 or C2)
        rough_labels (list): A list of simplified proficiency labels (A, B or C)
    '''
    # retrieve the ontology individual
    individual = get_ontology(indiv_path).load()

    # crop first element as it simply corresponds to the class occurence
    intention_list = individual.search(is_a=individual.NextIntention)
    intention = intention_list[len(intention_list)-5] # there are 4 sub-classes that are present at the end of the list

    # get the subclass of the inferred intention
    intention_types = [str(typ)[str(typ).find('.')+1:] for typ in intention.is_a]
    possible_inferred_types = ['LoadedPositiveIntention', 'LoadedNegativeIntention', 'NonLoadedPositiveIntention', 'NonLoadedNegativeIntention']
    for i in range(len(possible_inferred_types)):
        if possible_inferred_types[i] in intention_types:
            label = i
        
    individual.destroy()

    return label


def random_prefix(classes, ontology_concept, finetuning=True):
    '''
    Generate a random prefix based on the ontology concept.

    Args:
        classes (list): List of class labels.
        ontology_concept (str): Type of ontology (e.g., "ProficiencyLevel" or "NextIntention").
        finetuning (bool): If True, return a fine-tuned prefix for model training.

    Returns:
        tuple: A prefix string and the selected class index.
    '''
    # randomly select the class
    index = random.randint(0, len(classes) - 1)

    # select randomly the question to put in the prefix
    if ontology_concept == "ProficiencyLevel":
        question = random.choice(cefr_starters)
    elif ontology_concept == "NextIntention":
        # if the require class is loaded, then go with "emotive" starter. Otherwise go with factual starter
        if index < 3: 
            question = random.choice(emo_starters)
        else:
            question = random.choice(cefr_starters)
        
    # the prompt for the fine-tuned model
    if finetuning:
        prefix = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You will be asked a question and given an instruction to format your answer between brackets. Answer directly after the brackets, and start your answer with a word, not a figure. <|eot_id|><|start_header_id|>user<|end_header_id|>

        {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        [{ontology_concept}: {classes[index]}] 
        '''        

    # the prompt to use on the raw model (with explicit indication for fair comparison)
    else:
        prefix = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You need to generate a sentence that has a proficiency level of {classes[index]} according to the CEFR levels (Common European Framework of Reference). Generate the sentence right after this, without any additional text. <|start_header_id|>assistant<|end_header_id|>
        '''

    return prefix, index


def next_cefr_index_prefix(classes, cefr_index, user_prompt):
    '''
    Generates a prefix for the next CEFR index in conversation.

    Args:
        classes (list): List of class labels.
        cefr_index (int): Index of the CEFR level.
        user_prompt (str): User prompt.

    Returns:
        tuple: A prefix string and the CEFR index.
    '''
    prefix = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You will be given the user content and an instruction between brackets to format your answer properly. Answer directly after the brackets, and start your answer with a word, not a figure. Adapt the length of the answer to the instruction between brackets. <|eot_id|><|start_header_id|>user<|end_header_id|>

        {user_prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        [ProficiencyLevel: {classes[cefr_index]}] 
        '''

    return prefix, cefr_index

def next_emotion_index_prefix(classes, emotion_index, user_prompt):
    '''
    Generate a prefix for the next emotion index in conversation.

    Args:
        classes (list): List of class labels.
        emotion_index (int): Index of the emotion.
        user_prompt (str): User prompt.

    Returns:
        tuple: A prefix string and the emotion index.
    '''
    prefix = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You will be given the user content and an instruction between brackets to format your answer properly. Answer directly after the brackets, and start your answer with a word, not a figure. Adapt the length of the answer to the instruction between brackets. <|eot_id|><|start_header_id|>user<|end_header_id|>

        {user_prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        [NextIntention: {classes[emotion_index]}] 
        '''

    return prefix, emotion_index

def next_emotion_index_prefix_with_conv_history(classes, emotion_index, user_prompt, experiment):
    '''
    Generate a prefix for the next emotion index, taking into account the conversation history.

    Args:
        classes (list): List of class labels.
        emotion_index (int): Index of the emotion.
        user_prompt (str): User prompt.
        experiment (str): Experiment identifier.

    Returns:
        tuple: A prefix string and the emotion index.
    '''
    prefix = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You will be given the history of a user/agent conversation and an instruction between brackets to format your answer properly. Answer directly after the brackets, and start your answer with a word, not a figure. Adapt the length of the answer to the instruction between brackets. <|eot_id|><|start_header_id|>user<|end_header_id|>

        {format_user_history(user_prompt, experiment)} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        [ProficiencyLevel: {classes[emotion_index]}] 
        '''

    return prefix, emotion_index

def generate_from_random_prompts(args, ontology_concept, finetuning):
    '''
    Generate text from random prompts based on the specified ontology concept.

    Args:
        args (dict): Configuration dictionary containing 'tokenizer', 'model', 'device', 'batch_size', and 'max_new_tokens'.
        ontology_concept (str): The type of ontology concept ('ProficiencyLevel' or 'NextIntention').
        finetuning (bool): Indicates if the generation is for finetuning.

    Returns:
        tuple: (batch_labels, batch_generations)
            - batch_labels (list): Labels for each generated text.
            - batch_generations (list): Cleaned generated texts.
    '''
    # get the appropriate labels depending on the evaluation task
    if ontology_concept == "ProficiencyLevel":
        classes = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    elif ontology_concept == "NextIntention":
        classes = ['LoadedPositive', 'LoadedNegative', 'LoadedNeutral', 'NonLoadedPositive', 'NonLoadedNegative', 'NonLoadedNeutral']

    # initialize everything for generation
    tokenizer = args['tokenizer']
    model = args['model']
    batch_labels, batch_generations = [], []
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") # disable for qwen and phi since it does not exist
        ]
    
    # get prompts and labels for the whole batch
    prompts, labels = zip(*[random_prefix(classes, ontology_concept, finetuning) for _ in range(args['batch_size'])])
    batch_labels.extend(labels)

    # perform all batch generations (batched inference --> more efficient)
    inputs = tokenizer(list(prompts), return_tensors="pt", padding=True).to(args['device'])
    outputs = model.generate(**inputs, max_new_tokens=args['max_new_tokens'], pad_token_id=args['tokenizer'].eos_token_id, eos_token_id=terminators)
    results = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    for result in results:
        generation = result[result.find('<|end_header_id|>')+1:] 
        # remove any occurence of the proficiency level prefix to evaluate correctly the proficiency level of the generated text
        cleaned_generation = remove_bracketed_text(generation)
        batch_generations.append(cleaned_generation)

    return batch_labels, batch_generations 


def generate_from_next_cefr_index(args, cefr_index, user_prompt):
    '''
    Generate text for a specific CEFR index given a user prompt in a conversation.

    Args:
        args (dict): Configuration dictionary containing 'tokenizer', 'model', 'device', and 'max_new_tokens'.
        cefr_index (str): The CEFR level ('A1', 'A2', 'B1', 'B2', 'C1', 'C2').
        user_prompt (str): The user prompt to guide text generation.

    Returns:
        tuple: (label, generation)
            - label (str): The CEFR level.
            - generation (str): Generated text.
    '''
    classes = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    tokenizer = args['tokenizer']
    model = args['model']
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    # get prompts and labels for the whole batch
    prompt, label = next_cefr_index_prefix(classes, cefr_index, user_prompt)

    # Perform single utterance generation (not batched)
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(args['device'])
    
    outputs = model.generate(
        input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        max_new_tokens=args['max_new_tokens'], 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=terminators
    )

    # Decode only the newly generated tokens
    generated_tokens = outputs[:, input_tokens['input_ids'].shape[-1]:] if len(outputs.shape) > 1 else outputs
    results = tokenizer.decode(generated_tokens[0], skip_special_tokens=True) if len(outputs.shape) > 1 else tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Find the part of the result after the special token '<|end_header_id|>'
    generation = results[results.find('<|end_header_id|>')+1:] if '<|end_header_id|>' in results else results

    return label, generation 


def generate_from_next_intention(args, next_intention_index, user_prompt, experiment):
    '''
    Generate text for a specific next intention index given a user prompt in a conversation.

    Args:
        args (dict): Configuration dictionary containing 'tokenizer', 'model', 'device', and 'max_new_tokens'.
        next_intention_index (str): The next intention index ('LoadedPositive', 'LoadedNegative', 'NonLoadedPositive', 'NonLoadedNegative').
        user_prompt (str): The user prompt to guide text generation.
        experiment (str): The experiment context affecting intention interpretation.

    Returns:
        tuple: (label, generation)
            - label (str): The next intention label.
            - generation (str): Generated text.
    '''
    classes = ['LoadedPositive', 'LoadedNegative', 'NonLoadedPositive', 'NonLoadedNegative']
    tokenizer = args['tokenizer']
    model = args['model']
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    # get prompts and labels for the whole batch
    prompt, label = next_emotion_index_prefix_with_conv_history(classes, next_intention_index, user_prompt, experiment)

    # Perform single utterance generation (not batched)
    input_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(args['device'])
    
    outputs = model.generate(
        input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        max_new_tokens=args['max_new_tokens'], 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=terminators
    )

    # Decode only the newly generated tokens
    generated_tokens = outputs[:, input_tokens['input_ids'].shape[-1]:] if len(outputs.shape) > 1 else outputs
    results = tokenizer.decode(generated_tokens[0], skip_special_tokens=True) if len(outputs.shape) > 1 else tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Find the part of the result after the special token '<|end_header_id|>'
    generation = results[results.find('<|end_header_id|>')+1:] if '<|end_header_id|>' in results else results

    return label, generation 

def create_batch_individual(batch_index, file_path, experiment, repository, ontology_concept):
    '''
    This function runs the external shell script `call_ontology_cefr.sh` or `call_ontology_emo.sh` (depending on the use-case) to create an ontology individual for current batch.

    Parameters:
        batch_index (int): The index of the batch to be processed.
        file_path (str): The file path where the batch generated data has been previously stored.
        experiment (str): The name of the experiment.
    '''
    # define the command
    if ontology_concept == "ProficiencyLevel":
        command = ["../call_ontology_cefr.sh", str(batch_index), file_path, experiment, repository]
    elif ontology_concept == "NextIntention":
        command = ["../call_ontology_emo.sh", str(batch_index), file_path, experiment, repository]
    else:
        print("Error: unknown ontology concept. The accepted concepts are 'ProficiencyLevel' and 'NextIntention'.")

    # running the command using subprocess
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")