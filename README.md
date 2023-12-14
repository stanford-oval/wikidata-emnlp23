**Note: Repo under development. We have released all data and evaluation code. We are now working on releasing fine-tuned models.**

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

# Evaluation

To run evaluation on the dev set, prepare your prediction file in the same format as `predicted_results/best.json`, and supply it as the first parameter
for `execute_predictions("predicted_results/best.json", "WikiWebQuestions/dev.json")` in `eval_predictions.py`.
Then, run `python eval_predictions.py`.

If running on the test set, also change the second parameter to `"WikiWebQuestions/test.json"`

# Models & Inference

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

and then check out `run_refined.py` for an example of inferencing with it.

# Citation

If you have used data or code from this repository, please cite this paper:

```
@inproceedings{xu-etal-2023-fine,
    title = "Fine-tuned {LLM}s Know More, Hallucinate Less with Few-Shot Sequence-to-Sequence Semantic Parsing over {W}ikidata",
    author = "Xu, Silei  and
      Liu, Shicheng  and
      Culhane, Theo  and
      Pertseva, Elizaveta  and
      Wu, Meng-Hsi  and
      Semnani, Sina  and
      Lam, Monica",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.353",
    pages = "5778--5791",
    abstract = "While large language models (LLMs) can answer many questions correctly, they can also hallucinate and give wrong answers. Wikidata, with its over 12 billion facts, can be used to ground LLMs to improve their factuality. This paper presents WikiWebQuestions, a high-quality question answering benchmark for Wikidata. Ported over from WebQuestions for Freebase, it consists of real-world data with SPARQL annotation. This paper presents a few-shot sequence-to-sequence semantic parser for Wikidata. We modify SPARQL to use the unique domain and property names instead of their IDs. We train the parser to use either the results from an entity linker or mentions in the query. We fine-tune LLaMA by adding the few-shot training data to that used to fine-tune Alpaca. Our experimental results demonstrate the effectiveness of this methodology, establishing a strong baseline of 76{\%} and 65{\%} answer accuracy in the dev and test sets of WikiWebQuestions, respectively. By pairing our semantic parser with GPT-3, we combine verifiable results with qualified GPT-3 guesses to provide useful answers to 96{\%} of the questions in dev. We also show that our method outperforms the state-of-the-art for the QALD-7 Wikidata dataset by 3.6{\%} in F1 score.",
}
```
