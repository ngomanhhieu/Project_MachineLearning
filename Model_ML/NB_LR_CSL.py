import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# ---------------------
# Bitmap rules như bạn đã cho
bitmap_rules = {
    'sbytes': (lambda x: 1 if x >= 340000 else 0),
    'sloss': (lambda x: 1 if x >= 130 else 0),
    'sttl': (lambda x: 1 if x in [63, 255] else 0),
    'trans_depth': (lambda x: 0 if x in [0, 1] else 1),
    'swin': (lambda x: 1 if x in [0, 255] else 0),
    'dwin': (lambda x: 1 if x in [0, 255] else 0),
    'spkts': (lambda x: 1 if x >= 691 else 0),
    'dpkts': (lambda x: 1 if x >= 1433 else 0),
    'ct_src_dport_ltm': (lambda x: 1 if x >= 5 else 0),
    'ct_dst_sport_ltm': (lambda x: 1 if x >= 5 else 0),
    'ct_dst_src_ltm': (lambda x: 1 if x >= 45 else 0),
    'dttl': (lambda x: 1 if x == 253 else (0 if x in [29, 30, 31, 32] else -1)),
    'ct_state_ttl': (lambda x: 1 if x in [4, 5] else 0),
    'is_ftp_login': (lambda x: 1 if x == 2 else 0),
    'is_sm_ips_ports': (lambda x: 1 if x == 0 else 0),
    'dbytes': (lambda x: 1 if x >= 2000000 else 0),
    'dloss': (lambda x: 1 if x >= 1000 else 0),
    'smeansz': (lambda x: 1 if x >= 1500 else 0),
    'response_body_len': (lambda x: 1 if x >= 1000000 else 0),
    'ct_srv_src': (lambda x: 1 if x >= 50 else 0),
}

# Nếu có cột 'ct_flow_http_mthd' bạn thêm rule tương tự, nếu không có bỏ ra

# ---------------------
def apply_bitmap(df):
    bitmap = pd.DataFrame()
    for feature, func in bitmap_rules.items():
        if feature in df.columns:
            bitmap[feature + '_bit'] = df[feature].apply(func)
        else:
            print(f"Warning: Feature '{feature}' not found in dataframe columns.")
    # Xóa các cột có giá trị -1 nếu có
    bitmap = bitmap.loc[:, (bitmap != -1).all(axis=0)]
    return bitmap

# ---------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test, model_performance):
    print(f"Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    
    y_pred = model.predict(X_test)
    end_predict = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1s = f1_score(y_test, y_pred, average='weighted')
    
    print(f"[{name}] Accuracy: {accuracy:.2%}")
    print(f"[{name}] Recall: {recall:.2%}")
    print(f"[{name}] Precision: {precision:.2%}")
    print(f"[{name}] F1-Score: {f1s:.2%}")
    print(f"[{name}] time to train: {end_train - start:.2f} s")
    print(f"[{name}] time to predict: {end_predict - end_train:.2f} s")
    print(f"[{name}] total time: {end_predict - start:.2f} s\n")
    
    model_performance.loc[name] = [accuracy, recall, precision, f1s,
                                  end_train - start,
                                  end_predict - end_train,
                                  end_predict - start]

# ---------------------
def main():
    # Đọc dữ liệu (thay đường dẫn phù hợp với bạn)
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')

    # Giả sử label là cột 'label'
    y_train = train_df['label']
    y_test = test_df['label']

    # Áp dụng bitmap
    X_train = apply_bitmap(train_df)
    X_test = apply_bitmap(test_df)

    # Khởi tạo bảng kết quả
    model_performance = pd.DataFrame(columns=['accuracy','recall','precision','f1-score','train_time','predict_time','total_time'])

    # Model 1: Naive Bayes
    gnb = GaussianNB()
    evaluate_model("Naive Bayes", gnb, X_train, y_train, X_test, y_test, model_performance)

    # Model 2: Logistic Regression với max_iter tăng, solver 'lbfgs'
    logreg = LogisticRegression(max_iter=500, solver='lbfgs', n_jobs=-1)
    evaluate_model("Logistic Regression", logreg, X_train, y_train, X_test, y_test, model_performance)

    print("Summary:")
    print(model_performance)

if __name__ == "__main__":
    main()
