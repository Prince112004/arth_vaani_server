
import os
import subprocess
import tabula
import pandas as pd
from flask import Flask

app = Flask(__name__)


def ocr_pdf(input_pdf_path, output_pdf_path):
    try:
        subprocess.run(['ocrmypdf', '--skip-text', input_pdf_path, output_pdf_path], check=True)
        print(f"OCR applied successfully: {output_pdf_path}")
    except subprocess.CalledProcessError as e:
        print(f"OCR error: {e}")

def extract_tables_from_pdf(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        # Extract tables from ALL pages
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

        csv_files = []
        for i, table in enumerate(tables):
            output_csv = os.path.join(output_dir, f'table_page_{i+1}.csv')
            table.to_csv(output_csv, index=False)
            print(f"Extracted table saved: {output_csv}")
            csv_files.append(output_csv)

        return csv_file
    except Exception as e:
        print(f"Table extraction error: {e}")
        return []

def main():
    input_pdf = awa
    ocr_pdf_output = 'ocr_demo.pdf'
    tables_folder = 'extracted_tables'
    merged_csv = 'merged_tables.csv'

    # Step 1: Apply OCR to entire PDF
    ocr_pdf(input_pdf, ocr_pdf_output)

    # Step 2: Extract tables from all pages of OCR-processed PDF
    csv_files = extract_tables_from_pdf(ocr_pdf_output, tables_folder)
    if os.path.exists(ocr_pdf_output):
        os.remove(ocr_pdf_output)
        print(f"Deleted temporary OCR PDF: {ocr_pdf_output}")
if __name__ == "__main__":
    main()

import os
import shutil
import pandas as pd
import openai
import csv
import io

# 📂 Folder paths
input_csv_folder = 'extracted_tables'
output_text_folder = 'text_tables'
final_csv_folder = 'final_csv_output'

# Step 1️⃣: Delete old processed folders if they exist
for folder in [output_text_folder, final_csv_folder]:
    if os.path.exists(folder):
        shutil.rmtree(folder)  # Delete old folder
        print(f"Deleted old folder: {folder}")

# Step 2️⃣: Recreate necessary output folders
os.makedirs(output_text_folder, exist_ok=True)
os.makedirs(final_csv_folder, exist_ok=True)
os.makedirs(input_csv_folder, exist_ok=True)
# Step 3️⃣: Convert CSV files into plain text tables
for file_name in os.listdir(input_csv_folder):
    if file_name.endswith('.csv'):
        csv_path = os.path.join(input_csv_folder, file_name)

        # Read CSV
        df = pd.read_csv(csv_path)

        # Convert DataFrame to CSV-like plain text
        table_text = df.to_csv(index=False)

        # Save as .txt file
        text_file_name = file_name.replace('.csv', '.txt')
        text_output_path = os.path.join(output_text_folder, text_file_name)

        with open(text_output_path, 'w') as text_file:
            text_file.write(table_text)

        print(f"Processed and saved text file: {text_output_path}")

print("✅ All CSV tables converted to plain text successfully!")

import openai
import io
import os
import csv
import re

# 🔹 OpenAI API Key
api_key = os.environ.get('API_KEY')


if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Set it as an environment variable.")

# 🔹 Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# 🔹 Folder paths
text_folder = "text_tables"  # Folder containing .txt files
final_csv_folder = "final_csv_output"

# 🔹 Ensure output folder exists
os.makedirs(final_csv_folder, exist_ok=True)

def clean_gpt_output(file_path):
    # Read the content of the generated CSV file from GPT
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Check and remove the unwanted markers ('''csv at the start and ''' at the end)
    if content.startswith("```csv"):
        content = content.replace("```csv", '').strip()

    if content.endswith("```"):
        content = content.rstrip("```").strip()

    # Write the cleaned content back to a new CSV file
    cleaned_file_path = file_path.replace('.txt', '_cleaned.csv')
    with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
        cleaned_file.write(content)

    print(f"Cleaned CSV file saved as: {cleaned_file_path}")


def preprocess_text(raw_text):
    # 🔹 Remove unwanted commas at line start
    cleaned_text = re.sub(r'^\s*,', '', raw_text, flags=re.MULTILINE)

    # 🔹 Replace multiple spaces with commas (for CSV format)
    cleaned_text = re.sub(r'\s{2,}', ',', cleaned_text)

    return cleaned_text.strip()

def convert_text_to_csv(text_file_path):
    page_number = int(os.path.splitext(os.path.basename(text_file_path))[0].split('_')[-1])
    # 🔹 Check if the file exists
    if not os.path.exists(text_file_path):
        return

    # 🔹 Read the input text file
    with open(text_file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    print("✅ Successfully read input file!")

    # 🔹 Preprocess text before sending to GPT
    formatted_text = preprocess_text(raw_text)

    # 🔹 Construct OpenAI prompt
    prompt = f"""
    Convert the following structured text into a **properly formatted CSV**nothing should be extra even "```csv".
    - Merge multi-line age groups (like "15 years and" + "above") into one value.
    - Replace hyphens (-) with "to" (example: 15-29 → 15 to 29) and space between names with "_".
    - Capitalize the first letter of each word in column headers.
    - Remove any blank rows/columns or unwanted rows like Unnamed columns, (1),(2),....
    - Ensure numbers (percentages) are extracted cleanly without merging issues.

    Please name the table as 'name_page{page_number}' where the page number corresponds to the current page of the PDF.
    Here is the cleaned structured text:
    {formatted_text}
    """
    # 🔹 Call GPT Model
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use GPT-4o for better structured outputs
            messages=[
                {"role": "system", "content": "You are a structured data assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # 🔹 Extract structured CSV content safely
        csv_content = response.choices[0].message.content.strip()

        # 🔹 Check if response is empty
        if not csv_content:
            print("❌ No CSV content received from OpenAI! Skipping...")
            return

        # 🔹 Validate CSV structure
        rows = csv_content.split("\n")
        num_columns = len(rows[0].split(","))

        # 🔹 Convert into a CSV file safely
        csv_file_like = io.StringIO(csv_content)

        # 🔹 Define output path
        output_csv_path = os.path.join(final_csv_folder, os.path.basename(text_file_path).replace('.txt', '.csv'))

        # 🔹 Save properly formatted CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in csv.reader(csv_file_like):
                csv_writer.writerow(row)

        print(f"✅ Processed CSV saved as: {output_csv_path}")

    except openai.OpenAIError as e:
        print(f"❌ OpenAI API Error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

# 🔹 Process all .txt files in the "text_tables" folder
print("\n🔍 Checking available files in:", text_folder)
text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]

if not text_files:
    print("⚠️ No text files found! Make sure the folder contains .txt files.")
else:
    print(f"✅ Found {len(text_files)} text files. Processing...")

# 🔹 Loop through all .txt files and convert to CSV
for text_file in text_files:
    text_file_path = os.path.join(text_folder, text_file)
    convert_text_to_csv(text_file_path)

print("\n🎉 All text files processed successfully!")
# Delete intermediate folders
folders_to_delete = ['extracted_tables', 'text_tables']
for folder in folders_to_delete:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"🗑️ Deleted intermediate folder: {folder}")