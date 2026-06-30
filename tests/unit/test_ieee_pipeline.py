import pytest
import pandas as pd
import numpy as np
from src.pipelines.ieee_pipeline import IEEECISPipeline


@pytest.fixture
def base_config():
    return {
        "data": {
            "ieee": {
                "transaction_path": "dummy_tx.csv",
                "identity_path": "dummy_id.csv",
            }
        },
        "features": {
            "ieee_categorical": ["ProductCD", "card4", "card6"],
        },
    }


@pytest.fixture
def mock_raw_data():
    df_txn = pd.DataFrame(
        {
            "TransactionID": [100, 101, 102],
            "TransactionDT": [1000, 2000, 3000],
            "card1": [111, 111, 222],
            "TransactionAmt": [50.0, 150.0, 200.0],
            "ProductCD": ["W", "W", "H"],
            "card4": ["visa", "visa", "mastercard"],
            "card6": ["credit", "credit", "debit"],
            "V1": [1.0, np.nan, 3.0],
        }
    )
    df_id = pd.DataFrame(
        {
            "TransactionID": [100, 101],
            "id_01": [-5.0, np.nan],
        }
    )
    return df_txn, df_id


def test_pipeline_init(base_config):
    pipeline = IEEECISPipeline(base_config)
    assert pipeline.categorical_cols == ["ProductCD", "card4", "card6"]
    assert pipeline.ieee_config["transaction_path"] == "dummy_tx.csv"


def test_pipeline_preprocess(base_config, mock_raw_data):
    pipeline = IEEECISPipeline(base_config)
    df = pipeline.preprocess(mock_raw_data, fit=True)

    # Check merge on TransactionID (TransactionID 102 has id_01 as NaN because it's left join)
    assert len(df) == 3
    assert "id_01" in df.columns
    assert pd.isnull(df.loc[df["TransactionID"] == 102, "id_01"].values[0])

    # Check V-feature imputation and missingness indicator creation
    assert "V1_is_missing" in df.columns
    assert df.loc[1, "V1_is_missing"] == 1  # Originally NaN in mock_raw_data for row 1
    assert df.loc[0, "V1_is_missing"] == 0

    # Check categorical columns are encoded as integers
    for col in base_config["features"]["ieee_categorical"]:
        assert col in df.columns
        assert pd.api.types.is_integer_dtype(df[col])
        assert col in pipeline.label_encoders


def test_pipeline_preprocess_transform_mode(base_config, mock_raw_data):
    pipeline = IEEECISPipeline(base_config)
    # First fit
    pipeline.preprocess(mock_raw_data, fit=True)

    # Now transform (fit=False) with unseen category in R_emaildomain or ProductCD
    df_txn_new = pd.DataFrame(
        {
            "TransactionID": [103],
            "TransactionDT": [4000],
            "card1": [111],
            "TransactionAmt": [100.0],
            "ProductCD": ["NEW_VAL"],  # unseen
            "card4": ["visa"],
            "card6": ["credit"],
            "V1": [2.0],
        }
    )
    df_id_new = pd.DataFrame(
        {
            "TransactionID": [103],
            "id_01": [0.0],
        }
    )

    df_transformed = pipeline.preprocess((df_txn_new, df_id_new), fit=False)
    assert len(df_transformed) == 1
    # Check that NEW_VAL mapped to UNKNOWN and was processed without crashing
    assert df_transformed.loc[0, "ProductCD"] >= 0


def test_pipeline_extract_features(base_config):
    pipeline = IEEECISPipeline(base_config)
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 30],
            "card1": [111, 111, 111],
            "TransactionAmt": [10.0, 20.0, 30.0],
        }
    )
    featured_df = pipeline.extract_features(df)

    # Check cumulative count: expanding, shift(1), fillna(0)
    # Timeline for card1 (111):
    # - TxDT 10: count before = 0
    # - TxDT 20: count before = 1
    # - TxDT 30: count before = 2
    assert list(featured_df["card1_count_cumulative"]) == [0.0, 1.0, 2.0]

    # Check amount relative to historical mean spend
    # Mean for card 111 is (10 + 20 + 30) / 3 = 20
    # Ratio: amt / mean
    # 10 / 20 = 0.5, 20 / 20 = 1.0, 30 / 20 = 1.5
    assert np.allclose(list(featured_df["amount_to_mean_ratio"]), [0.5, 1.0, 1.5])


def test_run_pipeline(base_config, mock_raw_data):
    pipeline = IEEECISPipeline(base_config)
    # Mock load_raw to return mock_raw_data
    pipeline.load_raw = lambda: mock_raw_data

    final_df = pipeline.run_pipeline()
    assert len(final_df) == 3
    assert "card1_count_cumulative" in final_df.columns
    assert "amount_to_mean_ratio" in final_df.columns
    assert "V1_is_missing" in final_df.columns
