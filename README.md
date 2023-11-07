**Note: Repo under development. We have released all data and are now working on releasing models and evaluation code**

# Repo structure

- The WikiWebQuestions dataset can be found under directory `WikiWebQuestions`;

- Training data for all models published in our paper can be found under directory `training_data`;

- Prediction results for all models published in our paper can be found under directory `predicted_results`.

## Model names

The JSON file names correspond to the models in our paper in the following way:

- `best` refers to the best model in our paper, i.e., `WikiSP (ours)` in Section 6 tables;

- `no_mention_oracle` refers to the model named `No mentions, trained with Oracle NED ` in Section 6.2 (Table 2);

- `no_mention_refined` refers to the model named `No mentions, trained with ReFinED` in Section 6.2 (Table 2);

- `original_query_format` refers to the model named `Original SPARQL` in Section 6.3 (Table 3).

# Inference

**This section is under development.**

## Running NED model

Download the finetuned ReFiNED model by:

```
pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip 
mkdir -p <your_directory>
curl https://almond-static.stanford.edu/research/qald/refined-finetune/config.json -o <your_directory>/config.json
curl https://almond-static.stanford.edu/research/qald/refined-finetune/model.pt -o <your_directory>/model.pt
curl https://almond-static.stanford.edu/research/qald/refined-finetune/precomputed_entity_descriptions_emb_wikidata_33831487-300.np -o <your_directory>/precomputed_entity_descriptions_emb_wikidata_33831487-300.np
```

and then check out `run_refined.py` for an example of how to inference.
