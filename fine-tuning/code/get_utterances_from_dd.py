import datasets
from datasets import Dataset
from utils import *

def transform_to_utterance_wise(dataset_dict):
    """
    Transforms dialog-wise dailydialog dataset to an utterance-wise dataset.
    
    Each dialog in the input dataset is split into individual utterances, and each utterance is
    associated with its corresponding dialog act, emotion label, utterance index within the dialog,
    and dialog index within the dataset. The resulting dataset is structured so that each entry 
    corresponds to a single utterance.
    
    Args:
        dataset_dict (DatasetDict): A HuggingFace DatasetDict object with 'train', 'validation', and 'test' splits.
            Each split contains:
                - 'dialog': List of lists, where each inner list contains the utterances of a dialog.
                - 'act': List of lists, where each inner list contains the dialog acts corresponding to the utterances.
                - 'emotion': List of lists, where each inner list contains the emotion labels corresponding to the utterances.

    Returns:
        DatasetDict: A new DatasetDict object with 'train', 'validation', and 'test' splits.
            Each split contains:
                - 'utterance': Individual utterances extracted from the dialogs.
                - 'act': Corresponding dialog act for each utterance.
                - 'emotion': Corresponding emotion label for each utterance.
                - 'utt_idx': Index of the utterance within its dialog.
                - 'dial_idx': Index of the dialog within the dataset.
    """
    new_data = {'train': [], 'validation': [], 'test': []}
    
    for split in dataset_dict.keys():
        dialogs = dataset_dict[split]['dialog']
        acts = dataset_dict[split]['act']
        emotions = dataset_dict[split]['emotion']
        
        for dial_idx, (dialog, act_list, emotion_list) in enumerate(zip(dialogs, acts, emotions)):
            for utt_idx, (utterance, act, emotion) in enumerate(zip(dialog, act_list, emotion_list)):
                new_data[split].append({
                    'utterance': utterance.strip(),
                    'act': act,
                    'emotion': emotion,
                    'utt_idx': utt_idx,
                    'dial_idx': dial_idx
                })

    for split in new_data.keys():
        new_data[split] = Dataset.from_list(new_data[split])
    
    return DatasetDict(new_data)

if __name__ == '__main__':

    dd = datasets.load_dataset('daily_dialog')
    new_dd = transform_to_utterance_wise(dd)
    save_dataset(new_dd, 'utterances_dd')