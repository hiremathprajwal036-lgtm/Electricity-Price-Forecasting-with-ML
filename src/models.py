import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("xgboost not found - skipping xgboost model")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False
    print("tensorflow not available - skipping LSTM model")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def compute_metrics(y_true, y_pred, name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
    if name:
        print(f"{name}: MAE={mae:.3f} | RMSE={rmse:.3f} | MAPE={mape:.2f}% | R2={r2:.4f}")
    return metrics


class XGBoostForecaster:
    def __init__(self, params=None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("install xgboost first: pip install xgboost")
        self.params = params or {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self.feature_names = feature_names
        X_train_sc = self.scaler.fit_transform(X_train)
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_sc = self.scaler.transform(X_val)
            eval_set = [(X_val_sc, y_val)]
        self.model.fit(X_train_sc, y_train, eval_set=eval_set, verbose=False)
        print("xgboost training done")
        return self

    def predict(self, X):
        X_sc = self.scaler.transform(X)
        return self.model.predict(X_sc)

    def feature_importance(self):
        if self.feature_names is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(zip(self.feature_names, importances),
                           key=lambda x: x[1], reverse=True))

    def save(self, name="xgboost"):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self, os.path.join(MODELS_DIR, f"{name}.pkl"))
        print(f"saved -> models/{name}.pkl")

    @classmethod
    def load(cls, name="xgboost"):
        return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))


class RandomForestForecaster:
    def __init__(self, params=None):
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = RandomForestRegressor(**self.params)
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        self.feature_names = feature_names
        X_sc = self.scaler.fit_transform(X_train)
        self.model.fit(X_sc, y_train)
        print("random forest training done")
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def feature_importance(self):
        if self.feature_names is None:
            return {}
        importances = self.model.feature_importances_
        return dict(sorted(zip(self.feature_names, importances),
                           key=lambda x: x[1], reverse=True))

    def save(self, name="random_forest"):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self, os.path.join(MODELS_DIR, f"{name}.pkl"))
        print(f"saved -> models/{name}.pkl")

    @classmethod
    def load(cls, name="random_forest"):
        return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))


class LSTMForecaster:
    def __init__(self, sequence_len=24, units=64, dropout=0.2):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("install tensorflow first: pip install tensorflow")
        self.sequence_len = sequence_len
        self.units = units
        self.dropout = dropout
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None

    def _build_model(self, n_features):
        model = Sequential([
            LSTM(self.units, return_sequences=True,
                 input_shape=(self.sequence_len, n_features)),
            Dropout(self.dropout),
            BatchNormalization(),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def _make_sequences(self, X, y=None):
        Xs, ys = [], []
        for i in range(self.sequence_len, len(X)):
            Xs.append(X[i - self.sequence_len: i])
            if y is not None:
                ys.append(y[i])
        if y is not None:
            return np.array(Xs), np.array(ys)
        return np.array(Xs)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=30, batch_size=64):
        X_sc = self.scaler_X.fit_transform(X_train)
        y_sc = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        Xs, ys = self._make_sequences(X_sc, y_sc)

        self.model = self._build_model(n_features=X_train.shape[1])

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
        ]

        val_data = None
        if X_val is not None and y_val is not None:
            Xv_sc = self.scaler_X.transform(X_val)
            yv_sc = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            Xvs, yvs = self._make_sequences(Xv_sc, yv_sc)
            val_data = (Xvs, yvs)

        self.model.fit(
            Xs, ys,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0,
        )
        print("lstm training done")
        return self

    def predict(self, X):
        X_sc = self.scaler_X.transform(X)
        Xs = self._make_sequences(X_sc)
        if self.model is None:
         raise ValueError("LSTM model not trained")
         self.model.predict(X)
        preds_sc = self.model.predict(Xs, verbose=0)
        preds = self.scaler_y.inverse_transform(preds_sc).ravel()
        full_preds = np.full(len(X), np.nan)
        full_preds[self.sequence_len:] = preds
        return full_preds

    def save(self, name="lstm"):
        os.makedirs(MODELS_DIR, exist_ok=True)
        if self.model is None:
         raise ValueError("LSTM model not trained")
         self.model.save("models/lstm.h5")
         self.model.save(os.path.join(MODELS_DIR, f"{name}.h5"))
        joblib.dump(
            {"scaler_X": self.scaler_X, "scaler_y": self.scaler_y,
             "sequence_len": self.sequence_len},
            os.path.join(MODELS_DIR, f"{name}_meta.pkl"),
        )
        print(f"saved -> models/{name}.h5")


class NaiveForecaster:
    # simplest possible baseline: predict tomorrow = today (same hour)
    def predict(self, X_df, target="price"):
        return X_df[f"{target}_lag_24h"].values
