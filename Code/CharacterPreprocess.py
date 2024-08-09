import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# Load the CSV file
df = pd.read_csv('..\Data\HPCharactersData.csv')

# Function to extract 'Other Names' from the provided URL
def get_other_names(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the element with the itemprop attribute
        other_names_element = soup.find(attrs={"itemprop": "alternateName"})

        # Extract the text from the element if it exists
        if other_names_element:
            return other_names_element.get_text(strip=True)
        else:
            return ''
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return ''

# Apply the function to the 'Link' column and create a new column 'Other Names'
df['Other Names'] = df['Link'].apply(get_other_names)




# Function to process the 'Other Names' column
def process_other_names(text):
    # Ensure the input is a string
    if isinstance(text, str):
        # Remove any text inside parentheses including the parentheses
        text = re.sub(r'\(.*?\)', '', text)

        # Remove all quotes
        text = re.sub(r'["\']', '', text)

        # Replace semicolons with commas
        text = re.sub(r';', ',', text)

        # Find all remaining text that could be names separated by commas
        names = re.split(r',\s*', text)

        # Clean up extra whitespace and remove any empty entries
        cleaned_names = [name.strip() for name in names if name.strip()]

        # Join the cleaned names with a comma
        return ', '.join(cleaned_names)


# Apply the function to the 'Other Names' column
df['Other Names'] = df['Other Names'].apply(process_other_names)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_characters_file.csv', index=False)

print("Completed processing the Other Names.")