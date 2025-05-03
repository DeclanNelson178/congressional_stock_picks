from data.data_utils import load_standardized_feature_set
from sklearn.ensemble import IsolationForest


def get_isolation_model():
    df = load_standardized_feature_set(include_known=False)
    iso_forest = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    iso_forest.fit(df)
    return iso_forest

def run_isolation_forest():
    df = load_standardized_feature_set(include_known=False)
    iso_forest = get_isolation_model()
    anomaly_scores = iso_forest.decision_function(df)
    predictions = iso_forest.predict(df)
    df['anomaly_score'] = anomaly_scores
    df['anomaly_label'] = predictions
    return df