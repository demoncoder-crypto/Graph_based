import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import networkx as nx
import joblib
from neo4j import GraphDatabase
import logging

# Import configuration
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MODEL_PATH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Neo4j Interaction ---

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

def fetch_transactions(driver):
    """Fetches transaction data from Neo4j."""
    query = """
    MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
    RETURN a1.account_id AS sender,
           a2.account_id AS receiver,
           t.amount AS amount,
           t.timestamp AS timestamp,
           t.transaction_id AS transaction_id,
           // Optionally return account features if they exist
           a1.creation_date AS sender_creation_date,
           a2.creation_date AS receiver_creation_date
    LIMIT 10000 // Add a limit for safety during development
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            df = pd.DataFrame([record.data() for record in result])
            logging.info(f"Fetched {len(df)} transactions from Neo4j.")
            if df.empty:
                logging.warning("No transaction data found in Neo4j. Ensure data is loaded.")
            return df
    except Exception as e:
        logging.error(f"Error fetching transactions from Neo4j: {e}")
        return pd.DataFrame() # Return empty dataframe on error

# --- Feature Engineering ---

def engineer_basic_features(df):
    """Adds basic transaction and account-based features."""
    if df.empty:
        return df
    df['send_count'] = df.groupby('sender')['sender'].transform('size')
    df['receive_count'] = df.groupby('receiver')['receiver'].transform('size')
    df['total_sent'] = df.groupby('sender')['amount'].transform('sum')
    df['total_received'] = df.groupby('receiver')['amount'].transform('sum')
    # Add more features like time-based features if timestamp is available
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # df['hour_of_day'] = df['timestamp'].dt.hour
    return df

def engineer_graph_features(df):
    """Adds graph-based features using NetworkX."""
    if df.empty or 'sender' not in df.columns or 'receiver' not in df.columns:
        logging.warning("Skipping graph feature engineering due to missing data.")
        # Add empty columns to prevent errors downstream if needed
        for col in ['pagerank_sender', 'pagerank_receiver', 'in_degree_sender',
                    'out_degree_sender', 'in_degree_receiver', 'out_degree_receiver']:
            if col not in df.columns: df[col] = np.nan
        return df

    logging.info("Creating graph from transactions...")
    # Ensure node IDs are strings or hashable types for NetworkX
    df['sender'] = df['sender'].astype(str)
    df['receiver'] = df['receiver'].astype(str)
    G = nx.from_pandas_edgelist(
        df, source='sender', target='receiver', edge_attr=['amount', 'timestamp'], create_using=nx.DiGraph()
    )
    logging.info(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    try:
        logging.info("Calculating PageRank...")
        pagerank = nx.pagerank(G, alpha=0.85)
        df['pagerank_sender'] = df['sender'].map(pagerank)
        df['pagerank_receiver'] = df['receiver'].map(pagerank)
    except Exception as e:
        logging.warning(f"Could not calculate PageRank: {e}")
        df['pagerank_sender'] = np.nan
        df['pagerank_receiver'] = np.nan


    logging.info("Calculating degree centralities...")
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    df['in_degree_sender'] = df['sender'].map(in_degree)
    df['out_degree_sender'] = df['sender'].map(out_degree)
    df['in_degree_receiver'] = df['receiver'].map(in_degree)
    df['out_degree_receiver'] = df['receiver'].map(out_degree)

    # Add more graph features if desired (e.g., community detection, centrality measures)
    # try:
    #     communities = nx.community.greedy_modularity_communities(G.to_undirected()) # Example
    #     community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    #     df['community_sender'] = df['sender'].map(community_map)
    #     df['community_receiver'] = df['receiver'].map(community_map)
    # except Exception as e:
    #      logging.warning(f"Could not calculate communities: {e}")


    return df

# --- Model Training ---

def train_fraud_model(df):
    """Trains an Isolation Forest model for anomaly detection."""
    features = [
        'amount', 'send_count', 'receive_count', 'total_sent', 'total_received',
        'pagerank_sender', 'pagerank_receiver',
        'in_degree_sender', 'out_degree_sender',
        'in_degree_receiver', 'out_degree_receiver'
        # Add more engineered features here
    ]
    # Ensure all feature columns exist, adding missing ones with NaN if necessary
    for f in features:
        if f not in df.columns:
            logging.warning(f"Feature column '{f}' not found. Adding it with NaN values.")
            df[f] = np.nan

    X = df[features].copy()

    # Handle potential missing values (fill with 0 or use more sophisticated imputation)
    X = X.fillna(0) # Simple imputation for now

    if X.empty:
        logging.error("Cannot train model: Feature matrix is empty.")
        return None, None

    logging.info(f"Training Isolation Forest model on {X.shape[0]} samples and {X.shape[1]} features...")
    # Adjust parameters as needed
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, verbose=1)
    iso_forest.fit(X)
    logging.info("Model training complete.")

    # Predict anomalies on the training data (for evaluation or labeling)
    df['anomaly_score'] = iso_forest.decision_function(X)
    df['anomaly'] = iso_forest.predict(X) # -1 for anomalies, 1 for normal
    df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

    logging.info(f"Predicted anomalies: {(df['anomaly'] == -1).sum()} out of {len(df)}")

    return iso_forest, df


def save_model(model, path):
    """Saves the trained model to a file."""
    if model:
        try:
            joblib.dump(model, path)
            logging.info(f"Model saved successfully to {path}")
        except Exception as e:
            logging.error(f"Error saving model to {path}: {e}")
    else:
        logging.warning("No model to save.")

# --- Main Execution ---

def train_and_save_model():
    """Full pipeline: connect, fetch, feature engineer, train, and save."""
    driver = None
    try:
        driver = get_neo4j_driver()
        transactions_df = fetch_transactions(driver)

        if transactions_df.empty:
            logging.error("Exiting: No data fetched from Neo4j.")
            return

        transactions_df = engineer_basic_features(transactions_df)
        transactions_df = engineer_graph_features(transactions_df)

        # Display some info about the features
        logging.info("Feature engineering complete. Sample data with features:")
        logging.info(f"\n{transactions_df.head().to_string()}")
        logging.info(f"\nFeature descriptions:\n{transactions_df.describe().to_string()}")


        model, labeled_df = train_fraud_model(transactions_df)

        if model:
            save_model(model, MODEL_PATH)
            # Optionally save the labeled data
            # labeled_df.to_csv('data/labeled_transactions.csv', index=False)
            # logging.info("Labeled transaction data saved.")
        else:
             logging.error("Model training failed.")


    finally:
        if driver:
            driver.close()
            logging.info("Neo4j connection closed.")

if __name__ == "__main__":
    logging.info("Starting model training script...")
    train_and_save_model()
    logging.info("Model training script finished.")
