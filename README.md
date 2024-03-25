<p align="center">
    <img src="./images/Wikidata-logo-en.svg" width="100px" alt="Wikidata" />
    <h1 align="center">
        <b>Seq-to-Seq Semantic Parsing over Wikidata</b>
        <br>
        <a href="https://arxiv.org/abs/2305.14202">
            <img src="https://img.shields.io/badge/cs.CL-2305.14202-b31b1b" alt="arXiv">
        </a>
        <a href="https://github.com/stanford-oval/wikidata-emnlp23/stargazers">
            <img src="https://img.shields.io/github/stars/stanford-oval/wikidata-emnlp23?style=social" alt="Github Stars">
        </a>
    </h1>
</p>

# Repo structure

- The WikiWebQuestions dataset can be found under directory `WikiWebQuestions`;

- Training data for all models published in our paper can be found under directory `training_data`;

- Prediction results for all models published in our paper can be found under directory `predicted_results`.

# Published WikiSP (LLaMA) models

Two models in the paper are available on huggingface:

- https://huggingface.co/stanford-oval/llama-7b-wikiwebquestions: This model is trained on WikiWebquestions and the Stanford Alpaca dataset. In the paper, this is the `WikiSP (ours)` model in Section 6 tables.

- https://huggingface.co/stanford-oval/llama-7b-wikiwebquestions-qald7: This model is trained on both WikiWebquestions, Qald-7, and the Stanford Alpaca dataset. In the paper, this is the model in Section 7.

To download these models, you can use:

```
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="stanford-oval/llama-7b-wikiwebquestions-qald7", repo_type="model", local_dir="<PATH_TO_LOCAL_DIRECTORY>", local_dir_use_symlinks=False)'
```

Then, start the server in a separate terminal using [Huggingface's text-generation-inference library](https://github.com/huggingface/text-generation-inference/). We recommend using their provided Docker image given its ease of use. Run:

```
docker run --gpus all --shm-size 1g -p <port>:80 -v <PATH_TO_LOCAL_DIRECTORY>:/data ghcr.io/huggingface/text-generation-inference:1.3.4 --model-id /data/ --num-shard <number-of-gpus> --max-batch-total-tokens 4096
```

## Model training data names

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
