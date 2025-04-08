# Harry-Potter-DS-Project

Contributors: [Maya Cohen](https://github.com/mayacohen777), [Mika Littor](https://github.com/mika-littor), [Tomer Michaeli](https://github.com/TrueMichato)

## Project Description

This project analyzes character relationships in the Harry Potter books using data science. It includes text preprocessing, network analysis, community detection, and sentiment analysis.

For a more detailed overview of what we did and how, check out our [writeup](https://docs.google.com/document/d/1kD0nuFl37i0eRayPQkcXkEPqzDp6P1n2NzLitJ3yPVo/edit?usp=sharing).

## Live Demo

To see the final results in real time, visit the repository's [GitHub Pages](https://truemichato.github.io/Harry-Potter-DS-Project/dynamic_relationship_graph_1_10_sample.html) site.

## Code Folder

- [CharacterPreprocess.py](Code/CharacterPreprocess.py): Preprocesses character data from `HPCharactersData.csv`, extracting alternate names.
- [preprocess.py](Code/preprocess.py): Preprocesses book text from `harry_potter_books.csv`, resolving coreferences and tokenizing sentences.
- [dynamic_plots.py](Code/dynamic_plots.py): Creates dynamic, interactive plots of character connections using `plotly`.
- [plot_connections.py](Code/plot_connections.py): Generates static plots, performs page rank analysis, community detection, and evaluates community quality.
- [utils.py](Code/utils.py): Contains utility functions and path definitions.
- [dynamic_utils.py](Code/dynamic_utils.py): Contains utility functions for dynamic plots, including animation and GIF conversion.

## Data Folder

- [HPCharactersData.csv](Data/HPCharactersData.csv): Original character dataset.
- [character_names.csv](Data/character_names.csv): Processed character dataset with names, IDs, and alternate names.
- [harry_potter_books.csv](Data/harry_potter_books.csv): Original book dataset, divided by chapter.
- [harry_potter_books_preprocessed.csv](Data/harry_potter_books_preprocessed.csv): Preprocessed book dataset with resolved coreferences and cleaned text.
- [harry_potter_sentences.csv](Data/harry_potter_sentences.csv): Final dataset of sentences for analysis.

## Project Workflow and Evaluation

1. **Data Collection and Preprocessing:**

    - Collected character data and book content from CSV files, and did some manual verification to ensure the accuracy and reliability of the data.
    - Crawled web pages to enhance character data.
    - Handled special cases like the Weasley, Potter, and Malfoy families using regex.
    - Implemented coreference resolution using SpaCy and Coreferee, adjusting for name recognition issues.

2. **Graph Construction and Analysis:**
    - Constructed a simple connection graph based on character co-occurrence in sentences.
    - Applied PageRank to evaluate character importance and connectivity.
    - Implemented community detection using Louvain and Leiden algorithms.

3. **Sentiment Analysis:**
    - Performed sentiment analysis using TextBlob and CardiffNLP’s Twitter RoBERTa-base-sentiment model.
    - Compared model performance against expert tagging (serving as the experts in question, as we all have extensive knowledge in the subject matter. i.e., we are Potterheads).

4. **Dynamic Visualization:**
    - Created a dynamic PageRank graph to visualize character relationship changes throughout the series.

### Evaluation Metrics

- **Community Detection:** Assessed using partition coverage and performance. Louvain algorithm showed higher coverage (0.43) and performance (0.79) compared to Leiden.
- **Sentiment Analysis:**
  - TextBlob: Achieved high recall (0.96) and F1 score (0.85), balancing precision and recall.
  - CardiffNLP’s Twitter RoBERTa-base-sentiment: Higher accuracy (0.4) but lower recall (0.041) and F1 score (0.08).

## Future Work

If we ever have more time for this project, we would like to:

- In-depth sentiment analysis using models like BERT for emotion analysis.
- Incorporating domain-specific positive and negative words (mudblood was not categorized as either, which leaves room for improvement).
- Identifying specific relationship types (friends, enemies, lovers).

## Credits

- Harry Potter book series by J.K. Rowling, compiled into CSV format by [Gaston Sanchez](https://github.com/gastonstat/harry-potter-data).
- Harry Potter character data from kaggle user [Josè Roberto Canuto](https://www.kaggle.com/datasets/zez000/characters-in-harry-potter-books).
- Additional character data from the [Harry Potter Lexicon](https://www.hp-lexicon.org).
