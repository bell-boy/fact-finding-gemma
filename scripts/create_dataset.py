import torch
import pandas as pd

from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.auto import tqdm

gemma2b = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map="auto")
gemma2b.train()
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Define the SPARQL endpoint
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

n_athletes = 9999

# Set your SPARQL query
query = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?athlete ?athleteLabel ?sport ?sportLabel WHERE {
  {
    SELECT DISTINCT ?athlete ?sport WHERE {
      ?athlete wdt:P31 wd:Q5;         # Instance of human
               wdt:P641 wd:Q41323.    # American football
      BIND(wd:Q41323 AS ?sport)
    }
    LIMIT %d
  }
  UNION
  {
    SELECT DISTINCT ?athlete ?sport WHERE {
      ?athlete wdt:P31 wd:Q5;         # Instance of human
               wdt:P641 wd:Q5369.     # Baseball
      BIND(wd:Q5369 AS ?sport)
    }
    LIMIT %d
  }
  UNION
  {
    SELECT DISTINCT ?athlete ?sport WHERE {
      ?athlete wdt:P31 wd:Q5;         # Instance of human
               wdt:P641 wd:Q5372.     # Basketball
      BIND(wd:Q5372 AS ?sport)
    }
    LIMIT %d
  }

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
  }
}
""" % ((n_athletes//3), (n_athletes//3), (n_athletes//3))

# Set the query and return format
sparql.setQuery(query)
sparql.setReturnFormat(JSON)

# Execute the query and convert the results
results = sparql.query().convert()

prompt = "Fact: Tiger Woods plays the sport of golf.\nFact: {} plays the sport of"
data_rows = []


# Process the results
for result in tqdm(results["results"]["bindings"], position=0, leave=True):
    athlete = result["athleteLabel"]["value"]
    sport = result["sportLabel"]["value"]


    athlete_prompt = prompt.format(athlete)
    tokens = tokenizer(athlete_prompt, return_tensors="pt")

    last_name_index = tokens.input_ids.size(1) - 5

    logits = gemma2b(**tokens).logits
    probablities = logits.softmax(dim=-1)
    sport_prob = probablities[:, -1, :].max(dim=-1).values
    sport_token = probablities[:, -1, :].max(dim=-1).indices
    predicted_sport = tokenizer.convert_ids_to_tokens(sport_token)[0]

    new_row = {
      "Name": athlete,
      "Sport": sport,
      "Sport Token": predicted_sport,
      "Last Name Index": last_name_index
    }

    if sport_token.item() > .5:
      if sport == "American football" and predicted_sport == "▁football":
        data_rows.append(new_row)

      if sport == "baseball" and predicted_sport == "▁baseball":
        data_rows.append(new_row)

      if sport == "basketball" and predicted_sport == "▁basketball":
        data_rows.append(new_row)

    
dataset = pd.DataFrame(data_rows)
dataset.to_csv("data/athlete.csv", index=False)


