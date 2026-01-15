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
