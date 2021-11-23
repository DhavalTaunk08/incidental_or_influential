# incidental_or_influential

To run the script, take the take the script corresponding to the model you want to use. Just run the required cells for training and inference purpose. Directly running from start to end will also train the model will also perform inference.

Below list shows the script name and the corresponding model used:-

1. **analysis_and_preprocessing.ipynb** - This script shows few analysis done on the dataset.
2. **transformers_ire_bert_0.59903.ipynb** - This script uses BERT model to fine-tune the dataset.
3. **transformers_ire_scibert_0.58162.ipynb** - This script uses SciBERT model to fine-tune the dataset.
4. **transformers_ire_distilbert_0.55423.ipynb** - This script uses DistilBERT model to fine-tune the dataset.
5. **ULMfit_0.41223.ipynb** - This script uses ULMfit model to fine-tune the dataset.
6. **transformers_irebert_fulk_context.ipynb** - This script uses BERT model to fine-tune model on the full context dataset.
7. **SBERT/** - This directory contains the code to fine-tune SBERT on the dataset. 

## ULMfit
```
F1-macro score - 0.41223
```
ULMfit model link - [Link to model](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/dhaval_taunk_research_iiit_ac_in/EXzzKgg-pIFAhr_wTfXCo54BdZVkh6L3S96Y8iLpk4Xlwg?e=MzVfqy)

## BERT + Linear Layer
```
F1-macro score - 0.59903
```
BERT model link - [Link to model](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/dhaval_taunk_research_iiit_ac_in/Epn3YTWZeh5KnyhL2n0bmyYBzDs6p8zwy7re-7jvbNA5rw?e=Afu6L7)

## SciBERT + Linear Layer
```
F1-macro score - 0.58162
```
SciBERT model link - [Link to model](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/dhaval_taunk_research_iiit_ac_in/ElHT-v0hC7lMr6DFlx3odV0B0ghruzMWC-nelhY-aCU91w?e=SNtwi1)

## SBERT
```
F1-macro score - 0.54610
```

## BERT + BiLSTM
```
F1-macro score - 0.52670
```

## DistilBERT + Linear Layer
```
F1-macro score - 0.55243
```
