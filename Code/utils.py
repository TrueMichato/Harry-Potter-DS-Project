from transformers import pipeline
import torch
from datasets import Dataset





PATH_SENTENCES = r"..\Data\harry_potter_sentences.csv"
PATH_CHARACTERS = r"..\Data\character_names.csv"
PATH_NAMES_ID = r"..\Data\dict_names_id.pkl"
PATH_PAIR_COUNTS = r"..\Data\pair_counts.pkl"
PATH_PAIR_SENTENCES = r"..\Data\pair_sentences.pkl"
PATH_SET_SENTENCES = r"..\Data\set_sentences.pkl"
PATHS = {
    "sentences": PATH_SENTENCES,
    "characters": PATH_CHARACTERS,
    "names_id": PATH_NAMES_ID,
    "pair_counts": PATH_PAIR_COUNTS,
    "pair_sentences": PATH_PAIR_SENTENCES,
    "set_sentences": PATH_SET_SENTENCES,
}

PATH_SENTENCES_TOMER = "Data/harry_potter_sentences.csv"
PATH_CHARACTERS_TOMER = "Data/character_names.csv"
PATH_NAMES_ID_TOMER = "Data/dict_names_id.pkl"
PATH_PAIR_COUNTS_TOMER = "Data/pair_counts.pkl"
PATH_PAIR_SENTENCES_TOMER = "Data/pair_sentences.pkl"
PATH_SET_SENTENCES_TOMER = "Data/set_sentences.pkl"
PATHS_TOMER = {
    "sentences": PATH_SENTENCES_TOMER,
    "characters": PATH_CHARACTERS_TOMER,
    "names_id": PATH_NAMES_ID_TOMER,
    "pair_counts": PATH_PAIR_COUNTS_TOMER,
    "pair_sentences": PATH_PAIR_SENTENCES_TOMER,
    "set_sentences": PATH_SET_SENTENCES_TOMER,
}