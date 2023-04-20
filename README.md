# Attribution Methods for Explaining Transformers

## Install prerequisites and download datasets

This project was tested with Python 3.8.9.

The packages needed are in the <code>requirements.txt</code> file. They can be installed with 

<code>pip install -r requirements.txt</code>

The datasets are missing. They have to be downloaded and extracted by running the <code>download_and_extract_datasets.py</code>
script.

### SST

Now that the SST dataset is extracted, it needs to be preprocessed. To do that, run the <code>datasets_ours/sst/prepare.py</code> script from 
that directory (working dir = <code>datasets_ours/sst</code>).
It runs the other required scripts using <code>os.system</code> and assumes that Python scripts can be run using the <code>python</code> command.
Alternatively, you can specify the command, e. g. of you run python using the <code>python3</code> command.

```
python prepare.py --python_cmd python3 
```

For completeness, it runs the following scripts in this order:
* <code>datasets_ours/sst/create_phrase_sentiments.py</code>
* <code>datasets_ours/sst/create_sentences_tokens.py</code>
* <code>datasets_ours/sst/create_splits_csv.py</code>
* <code>datasets_ours/sst/extend_train_set.py</code>

Now the SST dataset is ready.

### CTDC

The CTDC dataset needs to be transformed into a format that can be easily processed. To do this,
run the following script with <code>datasets_ours/news</code> as a working directory

* <code>datasets_ours/news/generate_splits.py</code>

Now the dataset is ready.

## The SST Pipeline

We need to train a model, (*optional*) generate baselines, generate attributions, and process the attributions. To do this, we need to
run multiple scripts.

### Training an SST model
To fine-tune an SST model, we use the <code>run_glue.py</code> script. 

```
python run_glue.py --model_name_or_path <model> --output_dir sst_model_trained \
--do_train --do_eval --max_seq_length 128 \
--per_device_train_batch_size 8 --per_device_eval_batch_size 8
--learning_rate 1e-5 --num_train_epochs 2 \
--logging_strategy epoch \
--validation_file datasets_ours/sst/dev.csv --train_file datasets_ours/sst/train.csv \
--save_strategy epoch --evaluation_strategy epoch --seed -1
```

For the model, we use one of the following
* <code>bert-base-cased</code>
* <code>prajjwal1/bert-medium</code>
* <code>prajjwal1/bert-small</code>
* <code>prajjwal1/bert-mini</code>

### _(Optional) Generate custom baselines_ 
If we have a fine-tuned model <code>sst_model_trained</code>, we can generate baselines for Integrated Gradients. To
do this, we use the <code>generate_neutral_baselines_sst.py</code> script.

```
python generate_neutral_baselines_sst.py --model_folder sst_model_trained --output_folder sst_model_trained/baselines
```

### Generate attributions
Now that we have a trained model, and optionally custom baselines, we can generate attributions. To do this, we 
use the <code>create_attributions_sst.py</code> script. It has many options, some of which are explained below.

* <code>--output_dir</code> - Where to save the attributions
* <code>--model_path</code> - Which model to evaluate
* <code>--baselines_dir</code> - Where are the custom baselines (only if we have generated them in the previous step)
* <code>--use_prepared_hp</code> - If set to True, the script will use the hyperparameters we use in our final tests. This overrides any other options (like --sg_noise). 
* <code>--sg_noise</code> - Sets the noise_level of SmoothGRAD and SmoothGRAD x Input
* <code>--ig_baseline</code> - Which baseline to use for Integrated Gradients, one of <code>zero</code>, <code>pad</code>, <code>avg</code>, <code>custom</code> (if we have generated them)
* <code>--ks_baseline</code> - Which baseline to use for KernelSHAP, one of <code>pad</code>, <code>unk</code>, <code>mask</code>
* <code>--sg_noise_test</code> - If True, runs the noise test for SmoothGRAD
* <code>--ig_baseline_test</code> - If True, runs the baseline test for Integrated Gradients (requires --baselines_dir to be specified)
* <code>--ks_baseline_test</code> - If True, runs the baseline test for KernelSHAP

The <code>--use_prepared_hp</code> can be used to replicate our results.

The script checks if the model passed is a <code>BertForSequenceClassification</code> - if it is, the script loads the model through the Chefer et al. implementation
to enable the Chefer et al. relprop method to be evaluated. Otherwise, the relprop method is ignored.


### Evaluate attributions
Now that we have the attributions in a folder, e.g. <code>sst_attributions</code> folder, we can evaluate them.

The <code>process_attributions_sst.py</code> script does the evaluation. ***It supports only BERT models with the BERT-style tokenization***. It merges the tokens
to match SST's word-level annotations. Any other model will not work, and will produce incorrect results.

The usage is as follows

```
python process_attributions_sst.py \
--attrs_dir sst_attributions \
--output_file sst_metrics.csv \
--uncased False
```

The <code>--uncased</code> is important for the script to work. If the attributions were produced by an uncased model, set
<code>--uncased True</code>.

From the models we use, the following are uncased and need <code>--uncased True</code>
* <code>prajjwal1/bert-medium</code>
* <code>prajjwal1/bert-small</code>
* <code>prajjwal1/bert-mini</code>

This produces a CSV file with the results for each metric and method.


## The CTDC Pipeline

We need to perform k-fold cross validation of a model, then train the model on the entire train split, (optional) generate baselines, generate attributions,
and process the attributions. To do this, we need to run multiple scripts.

### K-Fold evaluation of a CTDC model
To initialize a model and use cross-validation, use the <code>train_ctdc_kfold.py</code> script. 
The script creates a <code></code> file with the initialized model so that each trained model has the same 
classification head. This file is important for the next step, don't delete it.

The model prints the evaluation results into console and contains commented wandb integration code.

```
python train_ctdc_kfold.py \
--model_name <model_name> \
--batch_size 2 \
--epochs 5 \
--lr 1e-5 \
--from_tf False
```

Set the <code>--from_tf</code> to <code>True</code> if the model is in the TensorFlow format on HuggingFace. 
The script automatically detects the <code>UWB-AIR/Czert-B-base-cased</code> model and sets <code>from_tf</code> to 
<code>True</code>.

We use the following models
* <code>UWB-AIR/Czert-B-base-cased</code>
* <code>Seznam/small-e-czech</code>
* <code>nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large</code>

### Train a CTDC model
When we have a model evaluated with the k-fold cross-validation script, we can train a model based on the 
<code><model_name>-base</code> file.

To do this, use the <code>train_ctdc.py</code> script.

```
python train_ctdc.py \
--model_name <model_name> \
--model_file <model_name>-base \
--output_dir ctdc-fine-tuned \
--batch_size 2 \
--lr 1e-5 \
--from_tf False
```

As before, the <code>UWB-AIR/Czert-B-base-cased</code> model is autodetected and <code>from_tf</code> is set appropriately.
The <code>model_name</code> has to be specified, because we need to a) save the model with <code>save_pretrained</code> and
b) load the tokenizer.
The <code>output_dir</code> will contain a pickled model for each training epoch, and a model loadable with <code>AutoModel.from_pretrained</code>.

### *(Optional) Generate custom baselines*
To generate custom baselines for Integrated Gradients, use the <code>generate_neutral_baselines_ctdc.py</code> script.

```
python generate_neutral_baselines_ctdc.py \
--model_folder ctdc-fine-tuned \
--output_dir ctdc-fine-tuned/baselines
```

### Generate attributions
Now that we have a trained model, and optionally custom baselines, we can generate attributions. To do this, we 
use the <code>create_attributions_ctdc.py</code> script. It has many options, some of which are explained below.

* <code>--output_dir</code> - Where to save the attributions
* <code>--model_path</code> - Which model to evaluate
* <code>--baselines_dir</code> - Where are the custom baselines (only if we have generated them in the previous step)
* <code>--use_prepared_hp</code> - If set to True, the script will use the hyperparameters we use in our final tests. This overrides any other options (like --sg_noise). 
* <code>--sg_noise</code> - Sets the noise_level of SmoothGRAD and SmoothGRAD x Input
* <code>--ig_baseline</code> - Which baseline to use for Integrated Gradients, one of <code>zero</code>, <code>pad</code>, <code>avg</code>, <code>custom</code> (if we have generated them)
* <code>--ks_baseline</code> - Which baseline to use for KernelSHAP, one of <code>pad</code>, <code>unk</code>, <code>mask</code>
* <code>--sg_noise_test</code> - If True, runs the noise test for SmoothGRAD
* <code>--ig_baseline_test</code> - If True, runs the baseline test for Integrated Gradients (requires --baselines_dir to be specified)
* <code>--ks_baseline_test</code> - If True, runs the baseline test for KernelSHAP

The <code>--use_prepared_hp</code> can be used to replicate our results.

The script checks if the model passed is a <code>BertForSequenceClassification</code> - if it is, the script loads the model through the Chefer et al. implementation
to enable the Chefer et al. relprop method to be evaluated. Otherwise, the relprop method is ignored.

### Evaluate attributions
To evaluate the attributions, use the <code>process_attributions_ctdc.py</code> script. Let's say we have the attributions in <code>attributions_ctdc</code> folder.

```
python process_attributions_ctdc.py \
--attrs_dir attributions_ctdc \
--output_file metrics_ctdc.csv 
```

This runs the script with the default settings, additionally you can specify the minimum PMI for keywords or the minimum word count for documents.

## Creating visualisations

A simple visualisation of an SST sample can be created with the <code>visualize.py</code> script.

```
python visualize.py \
--attrs_dir attributions_sst \
--indices 18,139,1038,1070 \
--output_dir html_visualisations
```

The <code>indices</code> are a comma-separated list (no spaces) which index the test split of the SST dataset (as sorted originally).
The <code>output_dir</code> will contain a <code><index>.html</code> file for each of the indices. This file contains visualisations
of attributions for all the methods in the <code>attrs_dir</code>. The script prints the letter-method mapping into console, but 
the visualisation have the following order

* Vanilla Gradients
* Gradients x Input
* Integrated Gradients, n=20
* Integrated Gradients, n=50
* Integrated Gradients, n=100
* SmoothGRAD, n=20
* SmoothGRAD, n=50
* SmoothGRAD, n=100
* SmoothGRAD x Input, n=20
* SmoothGRAD x Input, n=50
* SmoothGRAD x Input, n=100
* KernelSHAP, n=100
* KernelSHAP, n=200
* KernelSHAP, n=500
* (*Optional*) Chefer et al.


###



