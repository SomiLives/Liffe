from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
import openai
import mysql.connector
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': os.getenv('DB_PASSWORD', 'somnotho'),
    'database': 'LargeECommerceDB',
}

# OpenAI API key
openai_api_key = None
def get_db_connection():
    """Establish a database connection."""
    return mysql.connector.connect(**DB_CONFIG)

def fetch_metadata():
    """Fetch table metadata and relationships from the database."""
    metadata = {}
    try:
        with get_db_connection() as connection:
            with connection.cursor(dictionary=True) as cursor:
                # Fetch columns
                cursor.execute("""
                    SELECT TABLE_NAME, COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE();
                """)
                for row in cursor.fetchall():
                    table = row['TABLE_NAME']
                    column = row['COLUMN_NAME']
                    metadata.setdefault(table, {'columns': [], 'joins': []})['columns'].append(column)

                # Fetch relationships
                cursor.execute("""
                    SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL;
                """)
                for row in cursor.fetchall():
                    table = row['TABLE_NAME']
                    join = (row['COLUMN_NAME'], row['REFERENCED_TABLE_NAME'], row['REFERENCED_COLUMN_NAME'])
                    metadata[table]['joins'].append(join)

    except mysql.connector.Error as err:
        logger.error(f"Error fetching metadata: {err}")
    return metadata

def generate_sql_with_gpt(context, question):
    """Generate SQL query using OpenAI GPT."""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates SQL queries based on user questions and database schemas."},
            {"role": "user", "content": f"Context: {context}\n\nUser Question: {question}\n\nGuidelines:\n1. Generate a valid MySQL query for SQL sever database to answer the user's question.\n2. Use the provided context to construct queries that align with the database schema.\n3. If the context is insufficient to construct the query, respond with: 'I need more information about the database schema to generate an accurate query.'\n4. Avoid providing answers or explanations—respond only with the MySQL query.\n5. If the query involves selecting from a table, use 'WITH (NOLOCK)' to avoid locking the table.THIS IS THE MOST IMPORTANT.\n6. Write the response on the same line, no new lines in the response or tabs.Replace new lines with space.\n7. don't use the this character * for the columns, name all the necessary column names instead.\n8. Don't just use JOIN, rather use LEFT or INNER joins.\nThe query must work in an SQL server.THIS IS THE SECOND MOST IMPORTANT.\n\nSQL Query:"},
        ]

        # Call GPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-3.5-turbo" if gpt-4 is not available
            messages=messages,
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error generating SQL query: {e}")
        return "Failed to generate SQL query."

@app.route('/generate-sql', methods=['POST'])
def generate_sql():
    try:
        data = request.json
        question = data.get("question")
        if not question:
            return jsonify({"error": "Question cannot be empty."}), 400

        # Fetch metadata
        metadata = fetch_metadata()
        if not metadata:
            return jsonify({"error": "Failed to fetch metadata from the database."}), 500

        # Prepare context for the GPT model
        context = " ".join(  # Changed from "\n".join to " ".join to avoid newlines
            [f"Table: {table}, Columns: {', '.join(details['columns'])}" for table, details in metadata.items()]
        )

        # Generate SQL query using GPT
        sql_query = generate_sql_with_gpt(context, question)

        # Remove any newline characters from the SQL query
        clean_sql_query = sql_query.replace("\n", " ").strip()

        # Return the clean SQL query in the response
        return jsonify({"sql-query": clean_sql_query}), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
