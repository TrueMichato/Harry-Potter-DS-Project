import pandas as pd
import requests
from bs4 import BeautifulSoup

# # Load the CSV file
# df = pd.read_csv('..\Data\HPCharactersData.csv')
#
# # Function to extract 'Other Names' from the provided URL
# def get_other_names(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Check if the request was successful
#         soup = BeautifulSoup(response.content, 'html.parser')
#
#         # Extract the 'Other Names' using the provided CSS selector
#         other_names = soup.select_one(
#             '#content > div > div > section > div > div.row > div.col-md-4 > div.fact_box > div:nth-child(6) > span[itemprop="alternateName"]'
#         )
#
#         # Return the text if found, otherwise return an empty string
#         return other_names.text if other_names else ''
#     except Exception as e:
#         print(f"Error fetching data from {url}: {e}")
#         return ''
#
# # Apply the function to the 'Link' column and create a new column 'Other Names'
# df['Other Names'] = df['Link'].apply(get_other_names)
#
# # Save the updated DataFrame to a new CSV file
# df.to_csv('updated_characters_file.csv', index=False)
#
# print("Completed the extraction of Other Names.")





# Load the CSV file that was already created
df = pd.read_csv('updated_characters_file.csv')


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
    #     # Remove any text inside parentheses including the parentheses
    #     text = re.sub(r'\(.*?\)', '', text)
    #     # Regular expression to match only the text within quotes, ignoring text outside
    #     quotes = re.findall(r'["\']([^"\']*?)["\']', text)
    #
    #     if quotes:
    #         # Join all quoted names with a comma if there are multiple
    #         return ', '.join(quotes)
    #     else:
    #         # Return the text as is if there are no quotes
    #         return text.strip()
    # else:
    #     # If the value is not a string, return it as is or return an empty string
    #     return text

# Apply the function to the 'Other Names' column
df['Other Names'] = df['Other Names'].apply(process_other_names)

# Save the updated DataFrame to a new CSV file
df.to_csv('final_characters_file.csv', index=False)

print("Completed processing the Other Names.")