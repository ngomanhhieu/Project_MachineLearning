import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# === Bitmap encoding with user-defined rules ===
def bitmap_encode(df):
    bitmap_rules = {
        'sbytes': lambda x: 1 if x >= 1418 else 0,
        'dbytes': lambda x: 1 if x >= 1102 else 0,
        'sloss': lambda x: 1 if x > 0 else 0,
        'sttl': lambda x: 1 if x in [254, 255] else 0,
        'swin': lambda x: 1 if x == 255 else 0,
        'spkts': lambda x: 1 if x >= 12 else 0,
        'dpkts': lambda x: 1 if x >= 20 else 0,
        'trans_depth': lambda x: 0 if x in [0, 1] else 1,
        'ct_src_dport_ltm': lambda x: 1 if x >= 5 else 0,
        'ct_dst_sport_ltm': lambda x: 1 if x >= 5 else 0,
        'ct_dst_src_ltm': lambda x: 1 if x >= 45 else 0,
        'dttl': lambda x: 1 if x == 253 else 0,
        'ct_state_ttl': lambda x: 1 if x in [4, 5] else 0,
        'is_ftp_login': lambda x: 1 if x == 2 else 0,
        'ct_flow_http_mthd': lambda x: 1 if x in [0, 1, 2, 4, 16] else 0,
        'is_sm_ips_ports': lambda x: 1 if x == 0 else 0,
        'dloss': lambda x: 1 if x >= 1000 else 0,
        'smeansz': lambda x: 1 if x >= 1500 else 0,
        'response_body_len': lambda x: 1 if x >= 1000000 else 0,
        'ct_srv_src': lambda x: 1 if x >= 50 else 0
    }
    encoded_df = df.copy()
    for col in df.columns:
        if col in bitmap_rules:
            encoded_df[col] = df[col].apply(bitmap_rules[col])
        else:
            # Nếu cột không nằm trong rule thì bỏ qua hoặc dùng median 
            encoded_df[col] = df[col].apply(lambda x: 0)  # hoặc 0 tạm cho thống nhất
    return encoded_df

# === Preprocessing ===
def preprocess(df):
    le_dict = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    return df, le_dict

def apply_label_encoding(df, le_dict):
    df = df.copy()
    for col, le in le_dict.items():
        df[col] = df[col].astype(str)
        df = df[df[col].isin(le.classes_)]
        df[col] = le.transform(df[col])
    return df

# === Evaluate ===
def evaluate_model(name, model, X_train, X_test, y_train, y_test, perf_df):
    start = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    y_pred = model.predict(X_test)
    end_predict = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1s = f1_score(y_test, y_pred, average='weighted')

    print(f"{name} Results")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"F1-Score: {f1s:.2%}")
    print(f"Train Time: {end_train - start:.2f}s")
    print(f"Predict Time: {end_predict - end_train:.2f}s")
    print(f"Total Time: {end_predict - start:.2f}s\n")

    perf_df.loc[name] = [accuracy, recall, precision, f1s,
                         end_train - start,
                         end_predict - end_train,
                         end_predict - start]

def main():
    df_train = pd.read_csv("UNSW_NB15_training-set.csv")
    df_test = pd.read_csv("UNSW_NB15_testing-set.csv")

    df_train, le_dict = preprocess(df_train)
    df_test = apply_label_encoding(df_test, le_dict)

    X_train = df_train.drop("label", axis=1)
    y_train = df_train["label"]
    X_test = df_test.drop("label", axis=1)
    y_test = df_test["label"]

    # Encode bitmap 
    X_train_bitmap = bitmap_encode(X_train)
    X_test_bitmap = bitmap_encode(X_test)

    model_performance = pd.DataFrame(columns=["Accuracy", "Recall", "Precision", "F1-Score", "Train Time", "Predict Time", "Total Time"])

    # CSL - chỉ khác Ra-CSL ở thuật toán (nếu có)
    # Ví dụ CSL dùng LogisticRegression
    evaluate_model("CSL Logistic", LogisticRegression(max_iter=1000), X_train_bitmap, X_test_bitmap, y_train, y_test, model_performance)

    # Nếu cần thêm NB
    evaluate_model("CSL NB", BernoulliNB(), X_train_bitmap, X_test_bitmap, y_train, y_test, model_performance)

    print("\nModel Performance Summary:")
    print(model_performance)

if __name__ == "__main__":
    main()
