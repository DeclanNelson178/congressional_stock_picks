from models.deep_svdd import get_anomaly_scores, DeepSVDDNet, train_deep_svdd, run_deep_svdd
from models.deep_sad import run_deep_sad
from data.data_utils import load_standardized_feature_set, load_feature_set

if __name__=="__main__":
    df = run_deep_sad(.95)
    known = df.loc[df["known_instance"] == 1, "anomaly_score"].mean()
    unknown = df.loc[df["known_instance"] == 0, "anomaly_score"].mean()
    print(known / unknown)
    breakpoint()
    print("done")
