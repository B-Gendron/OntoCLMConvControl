# Conversational Control with Ontologies for Large Language Models: A Lightweight Framework for Constrained Generation

This repository contains the software material for the paper "Conversational Control with Ontologies for Large Language Models: A Lightweight Framework for Constrained Generation" accepted at KG & LLM @ LREC 2026. It gathers the scripts and objects that have been developed and used by the authors as part of this work. Due to the size limitation of the supplementary material, we could provide the adapter layers only for one use-case and we could not provide all the datasets and trained models mentioned in the paper. Consequently, the fine-tuning and the evaluation of large language models as presented in the paper can be performed only after all the auxiliary models have been trained and the necessary data have been retrieved. We apologize for the inconvenience and tried to put as much of the useful material as possible in the repository.

## Table of Contents

- [Content Overview](#contents)
- [Setup Guide](#quickstart)
- [Cite this Work](#cite)

<a name="contents"></a>
## Content Overview
The repository is organized as follows, containing four sub-folders:

```
├─── cefr_model
│   ├─── code
│   ├─── data
│   ├─── models
│   ├─── plots
│   └─── resources
├─── emotion_strategy
│   ├─── code
│   └─── logs
├─── fine-tuning
│   ├─── adapters
│   ├─── code
|   ├─── gen_quality_corpora
│   ├─── objects
│   ├─── rdf
│   └─── results
├─── onto-models
│   ├─── code
│   ├─── models
│   ├─── rdf
│   └─── resources
├─── README.md
├─── appendices.pdf
└─── requirements.txt
```

### `cefr_model`

This subfolder contains all the utils to train the decision tree classifier that classifies texts in their right Proficiency CEFR levels (A1, A2, B1, B2, C1, C2). The fitted decision tree that we used for further evaluation is stored in `models/best_clf.pkl`. The decision tree training script is `decision_tree.py`. In order to compare this classifier to some upper bounds, we tried to fit a Llama model (with a classification head) on the same task, in `llama_upper_bound.py`. Additionally, as we have a dataset provided with two expert annotations for each data sample, we compute the inter-annotator agreeement score in `annotators_upper_bound.py`.

### `emotion_strategy` 

This subfolder contains the code to fine-tune RoBERTa-based models on load and polarity classification. Load classification is a binary classification (the text is either loaded or non loaded), while polarity classification is a three-class classification (negative, neutral and positive polarities). These classifiers are fine-tuned respectively in `code/load_predictor.py` and `code/polarity_predictor.py`. Unfortunately, we couldn't provide the datasets in the software but a description of the common datasets gatherings is provided in the scripts. Then, the script `code/evaluation.py` is used to annotate the data dedicated to LLM fine-tuning with the classifiers predictions. Moreover, in the `logs` folder, we provide the logs of the best fine-tuning runs of the two classifiers.

### `onto-models`

This subfolders contains the ontologies and (normally, when enough space) the auxiliary models to evaluate the descriptor values on the textual content and incorporate this information in the ontology. Each utterance will be therefore seen as an ontology individual. For that, we use the `owlready2` python library that allows to create an ontology and call reasoners to perform inference on it. This part of the code is crucial as it allows to retrieve the inferred concepts that will then be used to annotate fine-tuning data, to evaluate generated content and to guide the implementation of conversational strategies. Regarding the Proficiency-Level Control use-case, there are two ontologies: `rdf/utterances_cefr.rdf` only for the one-shot generation evaluation and `rdf/controllable_conversation.rdf` for the conversation strategy. The first one is processed in `code/finetuning_batch_individuals_cefr.py` and the second one in `code/finetuning_individuals_cefr_conv.rdf`. For the second use case, the ontology building has been harmonized in order to use only one ontology (`rdf/emotion_strategy.rdf`) and one script (`finetuning_batch_individuals_emo.py`).

### `fine-tuning` 

This subfolder contains all the scripts to fine-tune and evaluate large language models on our ontology-guided controlled generation task. More details about this subfolder are provided in the next section.

In the `gen_quality_corpora` subfolder, there are all the generated contents used to compute the automated metrics and the BERT score ratio, which is achieved in the script `../code/compute_gen_scores.py`. The scores are given in two json files (for reference/reference and predictions/references scores): `./results/generation_quality_ref_ref.json` and `./results/generation_quality_results.json`.

### Appendices

We provide a PDF file that contains additional information mostly regarding the way we modeled descriptors and examples of textual contents for different proficiency levels. 

<a name="quickstart"></a>
## Setup Guide

Start by setting up all the required models and python modules to run the code:
```bash
cd conv_strategies_onto
python3 -m venv conv_strategies
source conv_strategies/bin/activate
# install the required libraries and their dependencies
pip3 install -r requirements.txt
# install SpaCy model for linguistic feature retrieval
python3 -m spacy download en_core_web_sm
```

The following subsections give examples of how to run the code for fine-tuning and evaluation, provided we have all the necessary data and auxiliary models aleardy fitted.

### Run Fine-Tuning

The fine-tuning script is `fine-tuning/code/lm_training.py`. Set the model to fine-tune using a valid HuggingFace path here:
```python
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
```
In the paper, the HuggingFace paths we used are the following:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

Then, also specify the dataset to use for fine-tuning. To run Proficiency-Level Control fine-tuning task, use:
```python
dataset = load_dataset('onto_2prefix_abc_descriptive_cefr_dd')
```
Elsewhere, to run Emotion Profile Control task, use:
```python
dataset = load_dataset('emo_label_wrapped')
```
still in `lm_training.py` script. 

Lastly, specify the path where the training checkpoints should be saved in the `output_dir` parameter of `TrainingArguments` object:
```python
training_args = TrainingArguments(
    output_dir="../my_checkpoints/mistral_emo",
    eval_strategy="epoch",
    learning_rate=5e-4,
    weight_decay=0.01,
    num_train_epochs=20,
    seed=42,
    eval_accumulation_steps=1,
    prediction_loss_only=True,
)
```

### Run Evaluation

The evaluation script is `fine-tuning/code/lm_evaluation.py`. Start by defining the model you want to evaluate and provide the corresponding checkpoint for the adapters, depending also on the task you want to evaluate:
```python
model_name = "meta-llama/Meta-Llama-3-8B-Instruct" # select the LLM to evaluate
adapter_model_name = "../adapters/llama3_cefr/checkpoint-165500"  # put the right checkpoint here
```

If you want to train the raw model, meaning without the adapters, simply comment this line:
```python
clm_model = PeftModel.from_pretrained(clm_model, adapter_model_name)
```

Set the parameter `conversation` to False for one-shot generation evaluation, and the True to start a conversation with the model. By default, in conversation setup, you will be able to prompt the model 5 times in the same conversation. If you want to proceed bigger conversation, just change the parameter `'eval_iters': 5`.

Lastly, before running the evaluation, check that the parameters in the last call are compliant with you evaluation objective. In one-shot generation, this will evaluate on Proficiency-Level Control task:
```python
evaluate(args, f'cefr_llama3', ontology_concept='ProficiencyLevel')
```
In conversation, this will start a conversation following the Emotion Profile Control strategy:
```python
evaluate_conv(args, f'emotion_conv_test', ontology_concept="NextIntention")
```
<a name="cite"></a>
## Citation

```bibtex
@inproceedings{gendron-etal-2026-conversational,
    title = "Conversational Control with Ontologies for Large Language Models: A Lightweight Framework for Constrained Generation",
    author = "Gendron, Barbara  and
      Guibon, Gael  and
      d{'}Aquin, Mathieu",
    editor = "S{\'e}rasset, Gilles  and
      Gkirtzou, Katerina  and
      Cochez, Michael  and
      Kalo, Jan-Christoph",
    booktitle = "Proceedings of the Knowledge Graphs and Large Language Models Workshop ({KG}-{LLM}) @ {LREC}26",
    month = may,
    year = "2026",
    address = "Palma, Mallorca (Spain)",
    publisher = "ELRA Language Resources Association (ELRA)",
    url = "https://aclanthology.org/2026.kallm-1.3/",
    doi = "10.63317/446x4ysrhfeq",
    pages = "20--35"
}
```
