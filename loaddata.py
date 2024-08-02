import csv
from tinydb import TinyDB

# Load the CSV file
def load_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader]
    return rows

# Write the rows to a TinyDB table
def write_to_table(table_name, rows, db_path):
    db = TinyDB(db_path)
    table = db.table(table_name)
    for row in rows:
        row['updated'] = False
        table.insert(row)

# Main function
def main():
    db_path = './data/db.json'  # replace with your desired TinyDB file path

    load('nouns', './data/Most-Used German - 1781 Nouns.csv', db_path)
    load('verbs', './data/Most-Used German - 1054 Verbs.csv', db_path)
    load('adverbs', './data/Most-Used German - 253 Adverbs.csv', db_path)
    load('adjectives', './data/Most-Used German - 648 Adjectives.csv', db_path)


def load(table_name, csv_file_path, db_path):
    rows = load_csv(csv_file_path)
    write_to_table(table_name, rows, db_path)

    

if __name__ == '__main__':
    main()
