# Natural Language Processing
## Core Concepts
- **NLP (Natural Language Processing)**: Field at the intersection of AI and linguistics enabling computers to understand human language.
- **Corpus** (plural: Corpora): A large collection of text or speech used for training NLP models. At the baseic level, it is a collection of sentences.
- **Document**: At basic level, a sentence can be considered a document. Corpus is a collection of documents
- **Tokenization**: Breaking text into smaller units (tokens) like words or sentences. At basic level, you tokenize a sentence into words (or even characters or sub-words/n-grams)
- **Vocabulary**: After tokenization, set of unique words (tokens) in the Corpus
- **Stop Words**: Common, low-meaning words (e.g., "the", "is") often removed during processing. After tokenizing, you remove these words before training the model
- **Out-of-Vocabulary (OOV)** Words: Words encountered during testing or deployment that weren't in the original training vocabulary. They pose a challenge for models.

## Text Pre-processing techniques
### STEMMING
- It reduces words to their "stem" or base form by stripping away affixes (prefixes and suffixes). For example, it removes "-ing", "-ed", "-s" etc from the words
- It is rule-based, heuristic process. Unlike lemmatization, it does not use a dictionary or understand context; it simply "chops off" the ends of words based on predefined rules.
- Because it is mechanical, the resulting stem is not always a valid dictionary word. 

  | Words | Stem |
  |---|---|
  | "running," "runner," "runs" | run |
  | "studies," "studying," "student" | studi |
  | Happiness, Happily | happi |
  | Universe, Universities | Univers |

- Common Stemming Algorithms
  - Different algorithms use different "aggressiveness" levels when stripping suffixes: 
    - Porter Stemmer: The most common and oldest (1980). It uses five sequential steps of rules to strip common English suffixes.
    - Snowball Stemmer (Porter2): An improved version of the Porter stemmer. It is more accurate and supports multiple languages besides English.
    - Lancaster Stemmer: Extremely aggressive and fast. It often results in stems that are very short and difficult to interpret. 

- Applications in NLP
  - Search Engines: Helps match a user's query (e.g., "investing") with documents containing variations like "invest" or "investment".
  - Sentiment Analysis: Groups variations of emotional words (e.g., "happy," "happily") to better determine the overall tone of a text.
  - Text Classification: Reduces the number of unique words (dimensionality reduction), which helps machine learning models learn patterns more efficiently.
  - 

- Key Challenges
  - Over-stemming: When the algorithm strips too much, merging words with different meanings (e.g., "universe" and "university" both becoming univers).
  - Under-stemming: When related words are not reduced to the same stem (e.g., "datum" and "data" becoming datu and dat).
  - Meaning Loss: Since it doesn't look at context, "better" might be stemmed to bet, which has a completely different meaning.

### LEMMATIZATION
- It is the process of reducing a word to its base or dictionary form, known as a lemma. 
- Unlike stemming, which blindly "chops off" word endings, lemmatization uses complex linguistic rules, dictionaries, and morphological analysis to ensure the resulting word is meaningful and grammatically correct.
- How Lemmatization Works ??
  - Lemmatization considers the context and the word's role in a sentence to determine its base form. This typically involves: 
    - Tokenization: Breaking text into individual words (tokens).
    - Part-of-Speech (POS) Tagging: Identifying whether a word is a noun, verb, adjective, etc.. This is critical because the same word can have different lemmas depending on its usage (e.g., "meeting" as a noun remains "meeting," but as a verb it becomes "meet").
    - Dictionary Lookup: Consulting a lexical database, such as WordNet, to find the exact base form.
<br/>

  | Original Word  | Lemma | Context/Reasoning |
  |---|---|---|
  | "better" | good | Identifies semantic root of the irregular adjective. |
  | "was" / "is" / "are" | be | Groups all forms of the verb "to be" together. |
  | "mice" | mouse | Resolves irregular plural forms. |
  | "studied" | study | Normalizes past tense to present base form. |
  | "ate" | eat | Resolves irregular past tense verbs. |

- Why Use Lemmatization?
  - Accuracy: It produces valid dictionary words, making it essential for complex tasks like chatbots and virtual assistants (e.g., Alexa, Siri) that need to understand intent.
  - Reduced Redundancy: By grouping "running," "ran," and "runs" under the single concept "run," it shrinks the vocabulary size and improves model efficiency.
  - Improved Search: Search engines like Google use lemmatization to match your queries with relevant documents that might use a different form of the same word.
  - Sentiment Analysis: It helps consistently score emotional words (e.g., "loved" and "loving" both become "love"), leading to more precise sentiment detection.

#### STEMMING VS LEMMATIZATION
| Feature  | Stemming | Lemmatization |
|---|---|---|
| Approach | Heuristic/Rule-based (Chops off ends) | Morphological analysis (Uses dictionary) |
| Speed | Very Fast | Slower (More complex due to dictionary lookups & context analysis) |
| Output Precision | May not be a real word (happi) | Always a valid dictionary word (happy) |
| Context | Ignores context | Considers context (e.g., Part of Speech) |
| Usecase | Use for high-speed indexing of massive datasets where exact word forms matter less | Use when meaning is critical (chatbots, translation) |

#### Code example - TOKENIZE, STOP WORD REMOVAL, STEMMING, LEMMATIZING
- **NLTK** (Natural Language Toolkit) & **Spacy** **libraries**  are widely used. NTLK has both STEMMIMG & LEMMATIZATION functions while Spacy has only Lemmatization.
- Spacy is more industrial grade and is highly used for Lemmatization because it automatically identifies Part-of-Speech (POS) tags to ensure the lemma is a valid dictionary word

    #### NLTK: Stemming with Stopword Removal
    ```python
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Necessary downloads
    nltk.download('punkt')
    nltk.download('stopwords')
    
    text = "The striped bats are hanging on their feet for the best view."
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    # Tokenize, remove stopwords, and stem
    tokens = word_tokenize(text.lower())
    stemmed_words = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    
    print(stemmed_words)
    # Output: ['stripe', 'bat', 'hang', 'feet', 'best', 'view']
    ```

    #### spaCy: Lemmatization with Stopword Removal
    ```python
    import spacy

    # Load the English model
    nlp = spacy.load("en_core_web_sm")
    
    text = "The striped bats are hanging on their feet for the best view."
    doc = nlp(text)
    
    # Filter out stopwords and punctuation, then lemmatize
    lemmatized_words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    print(lemmatized_words)
    # Output: ['stripe', 'bat', 'hang', 'foot', 'good', 'view']
    ```

### Parts-Of-Speech (POS) Tagging
- It assigns a grammatical category—such as noun, verb, or adjective—to each word in a sentence based on its definition and context
- This technique is essential for resolving ambiguity; for instance, identifying whether "book" is a noun ("read a book") or a verb ("book a flight").
- Common POS Tagsets
    - Most Python libraries use one of two standards:
      - Universal Tagset: A simplified set of 12 general tags like NOUN, VERB, and ADJ.
      - Penn Treebank Tagset: A detailed set of 36 tags that distinguish specifics, such as plural nouns (NNS) vs. singular nouns (NN), or past tense verbs (VBD) vs. base form verbs (VB)
     
    #### POS Tagging by NLTK
    NLTK uses the Penn Treebank tagset by default, providing high granularity.
    ```Python
    import nltk
    from nltk.tokenize import word_tokenize
    
    # Necessary downloads for 2026 pipelines
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    text = "The quick brown fox jumps over the lazy dog."
    tokens = word_tokenize(text)
    
    # Detailed Penn Treebank tagging
    tagged = nltk.pos_tag(tokens)
    print(tagged)
    # Output: [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ...]
    ```

    #### POS Tagging by spaCy
    industrial standard because it automatically identifies both coarse-grained (universal) and fine-grained (detailed) tags simultaneously.
    ```Python
    import spacy
    
    # Load the English model
    nlp = spacy.load("en_core_web_sm")
    
    text = "Time flies like an arrow."
    doc = nlp(text)
    
    # Print header
    print(f"{'Word':<10} | {'Universal':<10} | {'Detailed':<10} | {'Description'}")
    print("-" * 65)
    
    # Iterate and print each token's attributes
    for token in doc:
        description = spacy.explain(token.tag_)
        print(f"{token.text:<10} | {token.pos_:<10} | {token.tag_:<10} | {description}")
    ```

    Output-
    | Word | Universal (token.pos_) | Detailed (token.tag_) | Description (spacy.explain) |
    |---|---|---|---|
    | Time | NOUN | NN | noun, singular or mass |
    | flies | VERB | VBZ | verb, 3rd person singular present |
    | like | ADP | IN | conjunction, subordinating or preposition |
    | an | DET | DT | determiner |
    | arrow | NOUN | NN | noun, singular or mass |
    | . | PUNCT | . | punctuation mark, sentence closer |

### Named Entity Recognition (NER)
- NLP Task that automatically identifies and categorizes key information (entities) in unstructured text into predefined classes such as names of people, organizations, locations, dates, and more
- Most modern models recognize these standard categories: 
    - PERSON: Names of individuals (e.g., "John Doe").
    - ORG: Companies, agencies, or institutions (e.g., "Google").
    - GPE: Geopolitical entities like countries, cities, or states (e.g., "France").
    - DATE: Absolute or relative dates/periods (e.g., "January 15, 2026").
    - MONEY: Monetary values, including symbols (e.g., "$100 million").

    #### NER using spaCy
    spaCy is used for fast, production-ready pipelines on CPU
    ```Python
    import spacy

    # Load the optimized English model (v3.0+ architecture)
    nlp = spacy.load("en_core_web_sm")
    
    text = "Apple is looking at buying a U.K. startup for $1 billion in 2026."
    doc = nlp(text)
    
    print(f"{'Entity':<15} | {'Label':<10} | {'Description'}")
    print("-" * 50)
    
    # Extract entities from the .ents attribute
    for ent in doc.ents:
        print(f"{ent.text:<15} | {ent.label_:<10} | {spacy.explain(ent.label_)}")
    ```
    Output-
    | Entity | Label | Description |
    |---|---|---|
    | Apple | ORG | Companies, agencies, institutions |
    | U.K. | GPE | Countries, cities, states |
    | $1 billion | MONEY | Monetary values |
    | 2026 | DATE | Absolute or relative dates |

    #### High accuracy NER using HuggingFace Transformers
    For tasks requiring deeper contextual understanding (e.g., distinguishing "Apple" the fruit from "Apple" the company in complex sentences), transformer-based models like BERT or RoBERTa are preferred; Requires GPU
    ```Python
    from transformers import pipeline
    
    # Initialize the state-of-the-art NER pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    
    text = "Sundar Pichai, the CEO of Google, visited London today."
    entities = ner_pipeline(text)
    
    for ent in entities:
        print(f"Entity: {ent['word']} | Label: {ent['entity_group']} | Score: {ent['score']:.4f}")
    ```
    When using the Hugging Face ```dbmdz/bert-large-cased-finetuned-conll03-english``` model with the ```aggregation_strategy="simple"``` parameter in 2026, the model groups sub-tokens back into whole words and provides a confidence score for each entity. Because we used ```aggregation_strategy="simple"```, "Sundar Pichai" is returned as a single unit rather than separate tokens.
    | Entity | Label | Score | Start | End |
    |---|---|---|---|---|
    | Sundar Pichai | PER | 0.9994 | 0 | 13 |
    | Google | ORG | 0.9982 | 25 | 31 |
    | London | LOC | 0.9997 | 41 | 47 |
