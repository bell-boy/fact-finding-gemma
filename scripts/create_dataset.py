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

# Set your SPARQL query
query = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?athlete ?athleteLabel ?sport ?sportLabel WHERE {
  ?athlete wdt:P31 wd:Q5;         # Instance of human
           wdt:P641 ?sport.       # Sport played

  VALUES ?sport {wd:Q41323 wd:Q5369 wd:Q5372}  # Football, Baseball, Basketball

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
  }
}
LIMIT 10000
"""

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
    predicted_sport = tokenizer.convert_ids_to_tokens(probablities[:, -1, :].argmax(dim=-1))[0]
    #print(athlete, sport, predicted_sport)

    new_row = {
      "Name": athlete,
      "Sport": sport,
      "Last Name Index": last_name_index
    }

    if sport == "American football" and predicted_sport == "▁football":
      data_rows.append(new_row)

    if sport == "baseball" and predicted_sport == "▁baseball":
      data_rows.append(new_row)

    if sport == "basketball" and predicted_sport == "▁basketball":
      data_rows.append(new_row)

    
dataset = pd.DataFrame(data_rows)
dataset.to_csv("data/athlete.csv", index=False)


