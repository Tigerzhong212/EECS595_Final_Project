# Recreating AWS Lex through BERT’s Fine Tuning

Code for EECS 595 final project. (by Yuyue Tu, Zhen Zhong)


## Previous Work

Matthew Huggins, Sharifa Alghowinem, Sooyeon Jeong, Pedro Colon-Hernandez, Cynthia Breazeal, and Hae Won Park. 2021. Practical Guidelines for Intent Recognition: BERT with Minimal Training Data Evaluated in Real-World HRI Application. In Proceedings of the 2021 ACM/IEEE International Conference on Human-Robot Interaction (HRI '21). Association for Computing Machinery, New York, NY, USA, 341–350. DOI:https://doi.org/10.1145/3434073.3444671

## Setup
Requires Python 3

`pip install -r requirements.txt`

## Getting Started

Simple prediction:

`python pred.py --data_path "./snips/" --output_dir "./model_save_snips_1ep/" --input_path "./snips/sample_pred_input.json"`

Example command for training BERT model with corresponding accuracy:

`python train.py --data_path "./snips/"  --epochs 1 --batch_size 32 --output_dir "./model_save_snips_1ep_bert/ --per_intent 25"`

Example command for training RoBERTa model with corresponding accuracy:

`python train_roberta.py --data_path "./snips/"  --epochs 1 --batch_size 32 --output_dir "./model_save_snips_1ep_roberta/ --per_intent 25"`
