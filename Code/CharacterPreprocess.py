import pandas as pd
import requests
from bs4 import BeautifulSoup
import re


def fix_unique_names() -> None:
    df_characters = pd.read_csv("Data/character_names.csv")
    p_name = df_characters["Name"].apply(lambda x: x.split(" ")[0])

    unique_mask = p_name.duplicated(keep=False) == False

    df_characters.loc[unique_mask, "Other Names"] = (
        df_characters.loc[unique_mask, "Other Names"].fillna("")
        + df_characters.loc[unique_mask, "Other Names"].apply(
            lambda x: ", " if x else ""
        )
        + p_name[unique_mask]
    )

    df_characters["Other Names"] = df_characters["Other Names"].str.strip(", ")
    df_characters["Other Names"] = df_characters["Other Names"].str.replace(" , ", ", ")
    df_characters["Other Names"] = df_characters["Other Names"].fillna("")
    df_characters["Other Names"] = (
        df_characters["Other Names"].str.split(", ").apply(lambda x: ", ".join(set(x)))
    )

    df_characters.to_csv("Data/character_names.csv", index=False)


df = pd.read_csv('..\Data\HPCharactersData.csv')

def get_other_names(url) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')

        other_names_element = soup.find(attrs={"itemprop": "alternateName"})

        if other_names_element:
            return other_names_element.get_text(strip=True)
        else:
            return ''
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return ''

df['Other Names'] = df['Link'].apply(get_other_names)


def process_other_names(text):
    if isinstance(text, str):
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'["\']', '', text)
        text = re.sub(r';', ',', text)
        names = re.split(r',\s*', text)

        cleaned_names = [name.strip() for name in names if name.strip()]

        return ', '.join(cleaned_names)

df['Other Names'] = df['Other Names'].apply(process_other_names)


df['Id'] = df.index

df.to_csv('character_names.csv', index=False)