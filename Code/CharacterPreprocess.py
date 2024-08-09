import pandas as pd
import requests
from bs4 import BeautifulSoup

# Load the CSV file
df = pd.read_csv('..\Data\HPCharactersData.csv')

# Function to extract 'Other Names' from the provided URL
def get_other_names(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the 'Other Names' using the provided CSS selector
        other_names = soup.select_one(
            '#content > div > div > section > div > div.row > div.col-md-4 > div.fact_box > div:nth-child(6) > span[itemprop="alternateName"]'
        )

        # Return the text if found, otherwise return an empty string
        return other_names.text if other_names else ''
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return ''

# Apply the function to the 'Link' column and create a new column 'Other Names'
df['Other Names'] = df['Link'].apply(get_other_names)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_characters_file.csv', index=False)

print("Completed the extraction of Other Names.")
