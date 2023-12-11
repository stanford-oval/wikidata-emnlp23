
import requests

url = "https://www.wikidata.org/w/api.php"

countries = [
    'united states of america', 'usa', 'us'
    'united kingdom', 'gb',
    'canada', 'ca',
    'australia', 'au',
    'new zealand', 'nz',
    'south africa', 'za',
    'germany', 'de',
    'france', 'fr',
    'brazil', 'br',
    'argentina', 'ar',
    'china', 'cn',
    'india', 'in',
    'russia', 'ru',
    'japan', 'jp',
    'south korea', 'kr',
    'mexico', 'mx',
    'indonesia', 'id',
    'nigeria', 'ng',
    'egypt', 'eg',
    'saudi arabia', 'sa',
    'turkey', 'tr',
    'iran', 'ir'
]


states = [
    # United States
    'alabama', 'al',
    'alaska', 'ak',
    'arizona', 'az',
    'arkansas', 'ar',
    'california', 'ca',
    'colorado', 'co',
    'connecticut', 'ct',
    'delaware', 'de',
    'florida', 'fl',
    'georgia', 'ga',
    'hawaii', 'hi',
    'idaho', 'id',
    'illinois', 'il',
    'indiana', 'in',
    'iowa', 'ia',
    'kansas', 'ks',
    'kentucky', 'ky',
    'louisiana', 'la',
    'maine', 'me',
    'maryland', 'md',
    'massachusetts', 'ma',
    'michigan', 'mi',
    'minnesota', 'mn',
    'mississippi', 'ms',
    'missouri', 'mo',
    'montana', 'mt',
    'nebraska', 'ne',
    'nevada', 'nv',
    'new hampshire', 'nh',
    'new jersey', 'nj',
    'new mexico', 'nm',
    'new york', 'ny',
    'north carolina', 'nc',
    'north dakota', 'nd',
    'ohio', 'oh',
    'oklahoma', 'ok',
    'oregon', 'or',
    'pennsylvania', 'pa',
    'rhode island', 'ri',
    'south carolina', 'sc',
    'south dakota', 'sd',
    'tennessee', 'tn',
    'texas', 'tx',
    'utah', 'ut',
    'vermont', 'vt',
    'virginia', 'va',
    'washington', 'wa',
    'west virginia', 'wv',
    'wisconsin', 'wi',
    'wyoming', 'wy',
    
    # Canada
    'alberta', 'ab',
    'british columbia', 'bc',
    'manitoba', 'mb',
    'new brunswick', 'nb',
    'newfoundland and labrador', 'nl',
    'nova scotia', 'ns',
    'ontario', 'on',
    'prince edward island', 'pe',
    'quebec', 'qc',
    'saskatchewan', 'sk',
    'northwest territories', 'nt',
    'nunavut', 'nu',
    'yukon', 'yt'
]

states_and_abbreviations = {
    'al': 'alabama',
    'ak': 'alaska',
    'az': 'arizona',
    'ar': 'arkansas',
    'ca': 'california',
    'co': 'colorado',
    'ct': 'connecticut',
    'de': 'delaware',
    'fl': 'florida',
    'ga': 'georgia',
    'hi': 'hawaii',
    'id': 'idaho',
    'il': 'illinois',
    'in': 'indiana',
    'ia': 'iowa',
    'ks': 'kansas',
    'ky': 'kentucky',
    'la': 'louisiana',
    'me': 'maine',
    'md': 'maryland',
    'ma': 'massachusetts',
    'mi': 'michigan',
    'mn': 'minnesota',
    'ms': 'mississippi',
    'mo': 'missouri',
    'mt': 'montana',
    'ne': 'nebraska',
    'nv': 'nevada',
    'nh': 'new hampshire',
    'nj': 'new jersey',
    'nm': 'new mexico',
    'ny': 'new york',
    'nc': 'north carolina',
    'nd': 'north dakota',
    'oh': 'ohio',
    'ok': 'oklahoma',
    'or': 'oregon',
    'pa': 'pennsylvania',
    'ri': 'rhode island',
    'sc': 'south carolina',
    'sd': 'south dakota',
    'tn': 'tennessee',
    'tx': 'texas',
    'ut': 'utah',
    'vt': 'vermont',
    'va': 'virginia',
    'wa': 'washington',
    'wv': 'west virginia',
    'wi': 'wisconsin',
    'wy': 'wyoming',
}


def spans(mention):
    spans = []
    tokens = mention.split()
    for length in range(1, len(tokens) + 1):
        for index in range(len(tokens) - length + 1):
            span = ' '.join(tokens[index:index + length])
            spans.append(span)
    return spans


def search_span(span):
    candidates = []
    params = {
        "action": "wbsearchentities",
        "search": span,
        "language": "en",
        "limit": 5,
        "format": "json",
        "props": "description"
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    # Accessing the results
    results = data["search"]

    # Print the title of each result
    for result in results:
        candidates.append(result["id"])
        
    return candidates

def location_search(mention):
    mention = mention.replace('_', ' ')
    tokenized = mention.split()
    tokens = []
    for token in tokenized:
        if token in states_and_abbreviations:
            tokens.append(states_and_abbreviations[token])
        else:
            tokens.append(token)

    mention = ' '.join(tokens)
    candidates = search_span(mention)
    if len(candidates) > 0:
        return candidates[0]
    
    for state in states:
        if state in spans(mention):
            index = mention.index(state)
            candidates = search_span(mention[:index - 1] + ', ' + mention[index:])
            if len(candidates) > 0:
                return candidates[0]
            candidates = search_span(mention[:index-1])
            if len(candidates) > 0:
                return candidates[0]
            
    for country in countries:
        if country in spans(mention):
            index = mention.index(country)
            candidates = search_span(mention[:index - 1] + ', ' + mention[index:])
            if len(candidates) > 0:
                return candidates[0]
            candidates = search_span(mention[:index-1])
            if len(candidates) > 0:
                return candidates[0]

if __name__ == "__main__":
    print(location_search('anaheim_ca'))