from refined.inference.processor import Refined

refined = Refined.from_pretrained(
  model_name="models/refined",
  entity_set="wikidata",
  download_files=True,
  use_precomputed_descriptions=True
)

def refined_ned(utterance):
    spans = refined.process_text(utterance)
    output = set()
    for span in spans:
        if span.predicted_entity.wikidata_entity_id:
            qid = span.predicted_entity.wikidata_entity_id
            output.add(qid)
    
    return output
