import pandas as pd
from neo4j import GraphDatabase
import logging
import os

# Import configuration
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.csv')

def get_neo4j_driver():
    """Establishes a connection to the Neo4j database."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logging.info("Successfully connected to Neo4j.")
        return driver
    except Exception as e:
        logging.error(f"Failed to connect to Neo4j: {e}")
        raise

def create_constraints(tx):
    """Creates constraints for faster lookups and data integrity."""
    tx.run("CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:TRANSACTION) REQUIRE t.transaction_id IS UNIQUE")
    logging.info("Constraints ensured for Account.account_id and TRANSACTION.transaction_id")

def load_transactions_to_neo4j(driver, df):
    """Loads transaction data from a Pandas DataFrame into Neo4j."""
    # Cypher query to create nodes and relationships
    # MERGE ensures we don't create duplicate accounts
    # Using UNWIND is more efficient for batching
    query = """
    UNWIND $rows AS row
    // Merge sender account
    MERGE (sender:Account {account_id: row.sender_id})
    // Merge receiver account
    MERGE (receiver:Account {account_id: row.receiver_id})
    // Create transaction relationship
    MERGE (sender)-[t:TRANSACTION {transaction_id: row.transaction_id}]->(receiver)
    // Set transaction properties
    SET t.amount = toFloat(row.amount),
        t.timestamp = datetime(row.timestamp)
    """

    # Convert DataFrame to list of dictionaries for the driver
    data = df.to_dict('records')

    try:
        with driver.session(database="neo4j") as session: # Specify database if needed
            # Ensure constraints exist
            session.execute_write(create_constraints)

            # Run the import query
            logging.info(f"Loading {len(data)} transactions into Neo4j...")
            session.run(query, rows=data)
            logging.info("Successfully loaded data into Neo4j.")
    except Exception as e:
        logging.error(f"Error loading data into Neo4j: {e}")
        raise

def main():
    """Main function to read CSV and load data."""
    logging.info("Starting data loading script...")
    driver = None
    try:
        # Read data from CSV
        if not os.path.exists(DATA_FILE_PATH):
            logging.error(f"Data file not found: {DATA_FILE_PATH}")
            return

        logging.info(f"Reading data from {DATA_FILE_PATH}...")
        transactions_df = pd.read_csv(DATA_FILE_PATH)
        logging.info(f"Read {len(transactions_df)} records from CSV.")

        # Ensure required columns exist
        required_cols = ['transaction_id', 'sender_id', 'receiver_id', 'amount', 'timestamp']
        if not all(col in transactions_df.columns for col in required_cols):
            logging.error(f"CSV missing required columns: {required_cols}")
            return

        # Connect to Neo4j
        driver = get_neo4j_driver()

        # Load data
        load_transactions_to_neo4j(driver, transactions_df)

    except Exception as e:
        logging.error(f"An error occurred during data loading: {e}")
    finally:
        if driver:
            driver.close()
            logging.info("Neo4j connection closed.")
        logging.info("Data loading script finished.")

if __name__ == "__main__":
    main() 