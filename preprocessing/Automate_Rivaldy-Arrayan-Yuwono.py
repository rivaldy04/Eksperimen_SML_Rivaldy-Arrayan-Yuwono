from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from scipy.stats.mstats import winsorize
import pandas as pd

def preprocess_data(data, save_path, file_path):
    df = pd.read_csv(data)

    # Winsorize kolom 'depth' untuk mengatasi outlier ekstrem
    df['depth'] = winsorize(df['depth'], limits=[0.01, 0.01])

    # Menentukan fitur numerik yang akan diskalakan (sesuaikan dengan dataset Paduka)
    numeric_features = ['magnitude', 'depth', 'sig', 'nst', 'dmin', 'gap']

    # Pipeline untuk transformasi numerik menggunakan MinMaxScaler
    transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    # ColumnTransformer untuk menerapkan scaler hanya pada fitur numerik
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', transformer, numeric_features)
        ],
        remainder='passthrough'
    )

    # Terapkan transformasi ke data
    transformed_data = preprocessor.fit_transform(df)
    transformed_columns = numeric_features + [col for col in df.columns if col not in numeric_features]
    df_transformed = pd.DataFrame(transformed_data, columns=transformed_columns)
    df_transformed.to_csv(file_path, index=False)
    dump(preprocessor, save_path)

    print(f"✅ File hasil preprocessing disimpan di: {file_path}")
    print(f"✅ Pipeline disimpan di: {save_path}")

    return file_path

if __name__ == "__main__":
   result = preprocess_data(
    data="earthquake_data_tsunami.csv",
    save_path="preprocessing/preprocessor.joblib",
    file_path="preprocessing/earthquake_data_preprocessing.csv"
    )

