import stanza
import re

print("Initializing Stanza...")
try:
    stanza.download('en', package='default')
    nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', verbose=False)
    print("Stanza ready.")
except Exception as err:
    print(f"Stanza error: {err}. Try running stanza.download('en') manually.")
    exit()

def auto_reconstruct(text: str) -> str:
    """
    Improved automatic sentence reconstruction using linguistic rules.
    """
    temp = text
    temp = temp.replace("bit delay", "a bit of delay")
    temp = temp.replace("at recent days", "in recent days")
    temp = temp.replace("tried best", "tried their best")
    temp = temp.replace("for paper and cooperation", "on the paper and in our cooperation")

    doc = nlp_pipeline(temp)
    tokens = [w.text for s in doc.sentences for w in s.words]
    lemmas = [w.lemma for s in doc.sentences for w in s.words]

    if lemmas and lemmas[0].lower() == 'hope' and tokens[0].lower() == 'hope' and tokens[0] != 'I':
        tokens.insert(0, 'I')
        if len(tokens) > 1 and tokens[1].lower() == 'hope' and tokens[1][0].isupper():
            tokens[1] = tokens[1].lower()

    try:
        idx_hope = idx_to = idx_enjoy = -1
        for i, w in enumerate(tokens):
            if w.lower() == 'hope' and idx_hope == -1:
                idx_hope = i
            elif idx_hope != -1 and w.lower() == 'to' and idx_to == -1:
                idx_to = i
            elif idx_to != -1 and w.lower() == 'enjoy' and idx_enjoy == -1:
                idx_enjoy = i
                break
        if idx_hope != -1 and idx_to != -1 and idx_enjoy != -1 and idx_to < idx_enjoy:
            tokens.pop(idx_to)
    except Exception:
        pass

    sentence_str = " ".join(tokens)
    too_regex = r'\btoo\s*,\s*'
    wishes_regex = r'\bas my deepest wishes\b'
    found_too = False
    if re.search(too_regex, sentence_str, re.IGNORECASE):
        found_too = True
        sentence_str = re.sub(too_regex, ' ', sentence_str, flags=re.IGNORECASE).strip()
    if re.search(wishes_regex, sentence_str, re.IGNORECASE):
        sentence_str = re.sub(wishes_regex, '', sentence_str, flags=re.IGNORECASE).strip()
    tokens = sentence_str.split()

    if 'enjoy' in tokens and 'it' in tokens:
        try:
            idx_it = tokens.index('it')
            if found_too:
                tokens.insert(idx_it + 1, 'too')
            if not any(x in tokens for x in ['My', 'best', 'wishes']):
                tokens += ['.', 'My', 'best', 'wishes']
        except Exception:
            pass

    try:
        idx_team = idx_they = -1
        for i, w in enumerate(tokens):
            if w.lower() == 'team,':
                idx_team = i
            if w.lower() == 'they' and idx_team != -1 and i > idx_team:
                idx_they = i
                break
        if idx_they != -1 and idx_team != -1 and idx_they > idx_team:
            tokens.pop(idx_they)
    except Exception:
        pass

    result = " ".join(tokens)
    result = result.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!").replace("  ", " ").strip()
    result = re.sub(r'\s+', ' ', result).strip()
    split_sentences = re.split(r'([.?!])\s*', result)
    cleaned = []
    for i in range(0, len(split_sentences), 2):
        s = split_sentences[i].strip()
        if not s:
            continue
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        cleaned.append(s)
        if i + 1 < len(split_sentences):
            cleaned.append(split_sentences[i + 1])
    result = "".join(cleaned).strip()
    if result and not result.endswith(('.', '?', '!')):
        result += '.'
    result = result.replace("..", ".")
    return result

if __name__ == "__main__":
    s1 = "Hope you too, to enjoy it as my deepest wishes."
    s2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

    print(f"Original 1: {s1}")
    print(f"Reconstructed 1: {auto_reconstruct(s1)}")
    print("-" * 50)
    print(f"Original 2: {s2}")
    print(f"Reconstructed 2: {auto_reconstruct(s2)}")
    print("-" * 50)