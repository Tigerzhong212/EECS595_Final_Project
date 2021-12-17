import time
import datetime
import random
import os
import numpy as np
import pandas as pd
import pickle
import torch
import json
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
# from data import train_dataset, val_dataset, tokenizer, test_dataset, intent_dim, slots_dim, tag_values, metadata
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

from tqdm import tqdm
from params import params
from seqeval.metrics import f1_score

meta_path = params.data_path + "metadata.json"
with open(meta_path, "r") as f:
    metadata = json.load(f)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# intent_dim = metadata['intent_dim']
# slots_dim = metadata['slots_dim']
pad_len = metadata['pad_len']+1

# If there's a GPU available...
# if torch.cuda.is_available():

#     # Tell PyTorch to use the GPU.
#     device = torch.device("cuda")

#     # print('There are %d GPU(s) available.' % torch.cuda.device_count())

#     # print('We will use the GPU:', torch.cuda.get_device_name(0))

# # If not...
# else:
#     # print('No GPU available, using the CPU instead.')
#     device = torch.device("cpu")



# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
# batch_size = 32
# epochs = 1
# output_dir = './model_save/'

batch_size = params.batch_size
epochs = params.epochs
output_dir = params.output_dir


t0 = time.time()

# Load a trained model and vocabulary that you have fine-tuned
model = BertForSequenceClassification.from_pretrained(output_dir)
# tokenizer = BertTokenizer.from_pretrained(output_dir)

# Copy the model to the GPU.
# model.to(device)

model.eval()

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
# model = BertForTokenClassification.from_pretrained(
#     "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
#     num_labels = intent_dim+slots_dim, # The number of output labels--2 for binary classification.
#                     # You can increase this for multi-class tasks.
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
# )

# Tell pytorch to run this model on the GPU.
# if torch.cuda.is_available():
#     model.cuda()
#     model.cuda()

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
# optimizer = AdamW(model.parameters(),
#                   lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                   eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
#                 )


# Number of training epochs. The BERT authors recommend between 2 and 4.
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
# total_steps = len(train_dataloader) * epochs

# # Create the learning rate scheduler.
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = total_steps)


# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
# training_stats = []

# # Measure the total training time for the whole run.
# total_t0 = time.time()

# modified eval.py from line 230 to generate intent for input_sentence:

print("Load model time:", time.time() - t0)
t0 = time.time()

test_path = params.data_path + "test.pkl"

# tag_values = {0: "PAD"}

with open(test_path[:-8]+"intent_id_to_label.pkl", "rb") as f:
	intent_id_to_label = pickle.load(f)

# for k, v in intent_id_to_label.items():
# 	tag_values[k+1] = v


# input = "Hello dude"
# input = "turn on the light"
# input = "book the flight for me"
# input = params.input_sentence
input_ids = []
attention_masks = []
segment_ids = []

if params.input_path != "":
    with open(params.input_path, "r") as f:
        input_sentences = json.load(f)

# input_sentences = ["turn on the light", "book the flight for me", "will there be snow tomorrow in Ann Arbor", "play Adele's new album"]
for input in input_sentences:
    input_dict = tokenizer.encode_plus(
            input,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=pad_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
    )
    
    # Add the encoded sentence to the list.
    input_ids.append(input_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(input_dict['attention_mask'])

    segment_ids.append(input_dict['token_type_ids'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
segment_ids = torch.cat(segment_ids, dim=0)


test_labels = [0 for i in range(len(input_sentences))]
test_labels = torch.cat([torch.LongTensor([l]) for l in test_labels], dim=0)
# input_dataset = TensorDataset(input_dict['input_ids'], input_dict['attention_mask'], input_dict['token_type_ids'], test_labels)
input_dataset = TensorDataset(input_ids, attention_masks, segment_ids, test_labels)

input_dataloader = DataLoader(
            input_dataset, # The validation samples.
            sampler = SequentialSampler(input_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
)



for batch in input_dataloader:
    # Unpack this training batch from our dataloader.
    #
    # As we unpack the batch, we'll also copy each tensor to the GPU using
    # the `to` method.
    #
    # `batch` contains three pytorch tensors:
    #   [0]: input ids
    #   [1]: attention masks
    #   [2]: labels
    # b_input_ids = batch[0].to(device)
    # b_input_mask = batch[1].to(device)
    # b_segment_ids = batch[2].to(device)
    # b_labels = batch[3].to(device)

    
    b_input_ids = batch[0]
    b_input_mask = batch[1]
    b_segment_ids = batch[2]
    b_labels = batch[3]

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        # token_type_ids is the same as the "segment ids", which
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        (loss, logits) = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)

    # Accumulate the validation loss.
    # total_test_loss += loss.item()

    # Move logits and labels to CPU
    # logits = logits.detach().cpu().numpy()
    logits = logits.detach().numpy()
    # label_ids = b_labels.to('cpu').numpy()
 
    pred = np.argmax(logits, axis=1)
    print(pred)
    # print(type(pred))
    for i, id in enumerate(pred):
        print(input_sentences[i],":" intent_id_to_label[id])
    print("Prediction time (second):", time.time() - t0)
    