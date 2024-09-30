from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import networkx as nx
import joblib

# Replace with your Neo4j credentials
uri = "bolt://localhost:7687"
user = "neo4j"
password = "your_password"

driver = GraphDatabase.driver(uri, auth=(user, password))

def get_transactions(tx):
    query = """
    MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
    RETURN a1.account_id AS sender,
           a2.account_id AS receiver,
           t.amount AS amount,
           t.timestamp AS timestamp,
           t.transaction_id AS transaction_id
    """
    result = tx.run(query)
    return pd.DataFrame([record.data() for record in result])

with driver.session() as session:
    transactions_df = session.read_transaction(get_transactions)

def add_features(df):
    send_counts = df.groupby('sender').size().rename('send_count')
    receive_counts = df.groupby('receiver').size().rename('receive_count')
    df = df.join(send_counts, on='sender')
    df = df.join(receive_counts, on='receiver')
    send_amount = df.groupby('sender')['amount'].sum().rename('total_sent')
    receive_amount = df.groupby('receiver')['amount'].sum().rename('total_received')
    df = df.join(send_amount, on='sender')
    df = df.join(receive_amount, on='receiver')
    return df

transactions_df = add_features(transactions_df)

# Create a graph from transactions
G = nx.from_pandas_edgelist(
    transactions_df, source='sender', target='receiver', create_using=nx.DiGraph()
)

# Calculate PageRank as features
pagerank = nx.pagerank(G, alpha=0.85)
transactions_df['pagerank_sender'] = transactions_df['sender'].map(pagerank)
transactions_df['pagerank_receiver'] = transactions_df['receiver'].map(pagerank)

# Calculate centrality measures
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
transactions_df['in_degree_sender'] = transactions_df['sender'].map(in_degree)
transactions_df['out_degree_sender'] = transactions_df['sender'].map(out_degree)
transactions_df['in_degree_receiver'] = transactions_df['receiver'].map(in_degree)
transactions_df['out_degree_receiver'] = transactions_df['receiver'].map(out_degree)

# Select features and handle missing values
features = [
    'amount', 'send_count', 'receive_count', 'total_sent', 'total_received',
    'pagerank_sender', 'pagerank_receiver',
    'in_degree_sender', 'out_degree_sender',
    'in_degree_receiver', 'out_degree_receiver'
]
X = transactions_df[features].fillna(0)

# Train Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(X)

# Predict anomalies
transactions_df['anomaly_score'] = iso_forest.decision_function(X)
transactions_df['anomaly'] = iso_forest.predict(X)
transactions_df['anomaly_label'] = transactions_df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Save the trained model
joblib.dump(iso_forest, 'isolation_forest_model.joblib')

# Close the database connection
driver.close()
