"""End-to-end Privacy Shield demonstration script."""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
import torch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.utils.reproducibility import set_seed, get_device
from app.utils.logging import get_logger
from app.config import get_settings
from app.data.loader import DatasetLoader
from app.data.profiler import DatasetProfiler
from app.ml.preprocessing import PreprocessingPipeline
from app.ml.baseline_model import BaselineMLPConfig, BaselineMLP, BaselineTrainer
from app.ml.mia_attack import ModelInversionAttack
from app.ml.metrics import FraudDetectionMetrics
from app.privacy.input_protection import InputPrivacyModification
from app.privacy.local_dp import LocalDifferentialPrivacy
from app.privacy.compressive_privacy import CompressivePrivacy
from app.privacy.differential_privacy import DPSGDConfig, DPSGDTrainer

logger = get_logger("privacy_shield.demo")


def demo_privacy_shield():
    """Run complete Privacy Shield demonstration."""

    logger.info("=" * 80)
    logger.info("PRIVACY SHIELD - END-TO-END DEMONSTRATION")
    logger.info("=" * 80)

    # Settings
    settings = get_settings()
    device = get_device()
    set_seed(42)

    # ============================================================
    # PHASE 1: Load and Profile Dataset
    # ============================================================
    logger.info("\n[PHASE 1] Loading and profiling dataset...")

    loader = DatasetLoader("Base.csv", sample_size=10000)
    df = loader.load()
    logger.info(f"Dataset loaded: {df.shape}")

    profiler = DatasetProfiler(df)
    profile = profiler.profile_dataset()
    logger.info(profiler.generate_report())

    # ============================================================
    # PHASE 2: Data Preprocessing
    # ============================================================
    logger.info("\n[PHASE 2] Preprocessing data...")

    # Remove high-cardinality and ID-like columns
    identifiers_to_remove = ["device_distinct_emails_8w", "date_of_birth_distinct_emails_4w"]

    preprocessor = PreprocessingPipeline(
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
        remove_columns=identifiers_to_remove,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_and_transform(df)

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Target distribution - Train: {np.unique(y_train, return_counts=True)}")

    # ============================================================
    # PHASE 3: Train Baseline Model
    # ============================================================
    logger.info("\n[PHASE 3] Training baseline MLP...")

    baseline_config = BaselineMLPConfig(
        input_dim=X_train.shape[1],
        hidden_dim=128,
        num_hidden_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=20,
    )

    baseline_model = BaselineMLP(baseline_config)
    baseline_trainer = BaselineTrainer(baseline_model, baseline_config)

    baseline_history = baseline_trainer.train(X_train, y_train, X_val, y_val)

    baseline_val_loss, baseline_val_acc = baseline_trainer.evaluate(X_val, y_val)
    logger.info(f"Baseline Val Loss: {baseline_val_loss:.4f}, Acc: {baseline_val_acc:.4f}")

    # Evaluate on test set
    baseline_test_proba = baseline_trainer.predict(X_test)
    baseline_test_pred = baseline_trainer.predict_classes(X_test)

    baseline_metrics = FraudDetectionMetrics.calculate_metrics(
        y_test, baseline_test_proba, baseline_test_pred
    )
    logger.info(f"Baseline Test Metrics: F1={baseline_metrics['f1']:.4f}, ROC-AUC={baseline_metrics['roc_auc']:.4f}")

    # ============================================================
    # PHASE 4: Model Inversion Attack on Baseline
    # ============================================================
    logger.info("\n[PHASE 4] Running MIA on baseline model...")

    # Select a sensitive attribute to attack (customer_age)
    # We'll reconstruct it from features
    sensitive_attr_column = "customer_age"
    sensitive_attr_original = df[sensitive_attr_column].values[:len(y_test)]

    mia_baseline = ModelInversionAttack(baseline_model, device)
    baseline_attack_results = mia_baseline.run_attack(
        X_public=X_test,
        y_sensitive_true=sensitive_attr_original,
        sensitive_attribute_name=sensitive_attr_column,
        output_mode="probability",
        num_queries=min(1000, len(X_test)),
    )

    logger.info(f"Baseline MIA Attack Accuracy: {baseline_attack_results['attack_accuracy']:.4f}")

    # ============================================================
    # PHASE 5: Apply Privacy Modifications
    # ============================================================
    logger.info("\n[PHASE 5] Applying privacy-preserving data modification...")

    privacy_modifier = InputPrivacyModification()

    # Generalize age
    df_protected = df.copy()
    df_protected = privacy_modifier.generalize_age(df_protected, age_column="customer_age", bin_width=10)

    logger.info("Applied age generalization (10-year bins)")

    # ============================================================
    # PHASE 6: Apply Local Differential Privacy
    # ============================================================
    logger.info("\n[PHASE 6] Applying Local Differential Privacy...")

    ldp = LocalDifferentialPrivacy(epsilon=1.0)

    # Apply LDP to categorical features
    for col in preprocessor.categorical_features:
        if col in df_protected.columns:
            categories = df_protected[col].unique().tolist()
            df_protected[col] = ldp.randomized_response_array(df_protected[col].values, categories)

    logger.info("Applied LDP to categorical features")

    # ============================================================
    # PHASE 7: Apply Compressive Privacy
    # ============================================================
    logger.info("\n[PHASE 7] Applying Compressive Privacy...")

    # Re-preprocess protected data using the SAME split indices as baseline
    # to ensure train/val/test sets are aligned
    preprocessor_protected = PreprocessingPipeline(
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
        remove_columns=identifiers_to_remove,
    )

    # Extract the target before preprocessing
    y_target_protected = df_protected["fraud_bool"].values
    
    # Use the SAME train/val/test indices that were used in baseline preprocessing
    # This ensures consistent data splits and prevents LabelEncoder issues
    # Get original indices from preprocessor
    X_all_p = df_protected.drop("fraud_bool", axis=1)
    
    # Recreate the same split using the preprocessor's random seed
    from sklearn.model_selection import train_test_split as tts
    indices = np.arange(len(X_all_p))
    train_idx, temp_idx = tts(indices, test_size=0.30, random_state=42)
    val_idx, test_idx = tts(temp_idx, test_size=0.5, random_state=42)
    
    X_train_p = X_all_p.iloc[train_idx]
    X_val_p = X_all_p.iloc[val_idx]
    X_test_p = X_all_p.iloc[test_idx]
    y_train_p = y_target_protected[train_idx]
    y_val_p = y_target_protected[val_idx]
    y_test_p = y_target_protected[test_idx]
    
    # Now encode categorical features using a fresh encoder fit on training data
    # First, identify all categorical features
    categorical_cols = [col for col in X_train_p.columns if X_train_p[col].dtype == 'object']
    
    if categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        label_encoders_p = {}
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on train data only
            le.fit(X_train_p[col].astype(str))
            label_encoders_p[col] = le
            
            # Transform all splits - handle unknown by fitting on all data if needed
            try:
                X_train_p[col] = le.transform(X_train_p[col].astype(str))
                X_val_p[col] = le.transform(X_val_p[col].astype(str))
                X_test_p[col] = le.transform(X_test_p[col].astype(str))
            except ValueError as e:
                # If we get unknown labels, refit on all data
                logger.warning(f"Unknown category in {col}, refitting on all data")
                le.fit(pd.concat([X_train_p, X_val_p, X_test_p])[col].astype(str))
                X_train_p[col] = le.transform(X_train_p[col].astype(str))
                X_val_p[col] = le.transform(X_val_p[col].astype(str))
                X_test_p[col] = le.transform(X_test_p[col].astype(str))
    
    # Scale numerical features
    numerical_cols = [col for col in X_train_p.columns if X_train_p[col].dtype != 'object']
    if numerical_cols:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train_p[numerical_cols])
        X_train_p[numerical_cols] = scaler.transform(X_train_p[numerical_cols])
        X_val_p[numerical_cols] = scaler.transform(X_val_p[numerical_cols])
        X_test_p[numerical_cols] = scaler.transform(X_test_p[numerical_cols])

    # Apply PCA compression
    compressor = CompressivePrivacy(compression_ratio=0.8)
    X_train_comp = compressor.fit_transform(X_train_p)
    X_val_comp = compressor.transform(X_val_p)
    X_test_comp = compressor.transform(X_test_p)

    comp_stats = compressor.get_compression_stats()
    logger.info(
        f"Compression: {comp_stats['original_dim']} -> {comp_stats['compressed_dim']} "
        f"(ratio={comp_stats['compression_ratio']:.2%})"
    )

    # ============================================================
    # PHASE 8: Train DP-SGD Model
    # ============================================================
    logger.info("\n[PHASE 8] Training DP-SGD model...")

    dp_config = DPSGDConfig(
        input_dim=X_train_comp.shape[1],
        hidden_dim=64,
        num_hidden_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=20,
        epsilon=1.0,
        delta=1e-6,
        max_grad_norm=1.0,
        noise_multiplier=0.5,
    )

    dp_model = BaselineMLP(dp_config)
    dp_trainer = DPSGDTrainer(dp_model, dp_config)

    dp_history = dp_trainer.train(X_train_comp, y_train, X_val_comp, y_val)

    dp_val_loss, dp_val_acc = dp_trainer.evaluate(X_val_comp, y_val)
    logger.info(f"DP-SGD Val Loss: {dp_val_loss:.4f}, Acc: {dp_val_acc:.4f}")

    # Evaluate on test set
    dp_test_proba = dp_trainer.predict(X_test_comp)
    dp_test_pred = dp_trainer.predict_classes(X_test_comp)

    dp_metrics = FraudDetectionMetrics.calculate_metrics(y_test, dp_test_proba, dp_test_pred)
    logger.info(f"DP-SGD Test Metrics: F1={dp_metrics['f1']:.4f}, ROC-AUC={dp_metrics['roc_auc']:.4f}")

    privacy_params = dp_trainer.get_privacy_params()
    logger.info(f"Privacy Parameters: epsilon_spent={privacy_params['epsilon_spent']:.4f}")

    # ============================================================
    # PHASE 9: MIA on Protected Model
    # ============================================================
    logger.info("\n[PHASE 9] Running MIA on protected model...")

    mia_protected = ModelInversionAttack(dp_model, device)
    protected_attack_results = mia_protected.run_attack(
        X_public=X_test_comp,
        y_sensitive_true=sensitive_attr_original,
        sensitive_attribute_name=sensitive_attr_column,
        output_mode="probability",
        num_queries=min(1000, len(X_test_comp)),
    )

    logger.info(f"Protected MIA Attack Accuracy: {protected_attack_results['attack_accuracy']:.4f}")

    # ============================================================
    # PHASE 10: Comparison and Results
    # ============================================================
    logger.info("\n[PHASE 10] Privacy-Utility Trade-off Analysis...")

    comparison = FraudDetectionMetrics.compare_models(baseline_metrics, dp_metrics)

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    logger.info("\n--- Fraud Detection Utility ---")
    logger.info(f"Baseline F1:      {baseline_metrics['f1']:.4f}")
    logger.info(f"Protected F1:     {dp_metrics['f1']:.4f}")
    logger.info(f"Utility Change:   {comparison['differences']['f1']['percent']:+.2f}%")

    logger.info("\n--- Privacy Leakage (MIA Attack) ---")
    logger.info(f"Baseline MIA Acc:  {baseline_attack_results['attack_accuracy']:.4f}")
    logger.info(f"Protected MIA Acc: {protected_attack_results['attack_accuracy']:.4f}")
    privacy_improvement = (
        (baseline_attack_results["attack_accuracy"] - protected_attack_results["attack_accuracy"])
        / baseline_attack_results["attack_accuracy"]
        * 100
    )
    logger.info(f"Privacy Improvement: {privacy_improvement:+.2f}%")

    logger.info("\n--- Formal Privacy Guarantee (DP) ---")
    logger.info(f"epsilon_spent: {privacy_params['epsilon_spent']:.4f}")
    logger.info(f"delta: {privacy_params['delta']}")

    # ============================================================
    # Export Results
    # ============================================================
    logger.info("\n[PHASE 11] Saving results...")

    results = {
        "dataset": {
            "shape": df.shape,
            "fraud_rate": float(df["fraud_bool"].sum() / len(df)),
        },
        "baseline": {
            "metrics": baseline_metrics,
            "mia_attack": baseline_attack_results,
        },
        "protected": {
            "metrics": dp_metrics,
            "mia_attack": protected_attack_results,
            "privacy_params": privacy_params,
        },
        "comparison": {
            "utility_change_f1": comparison["differences"]["f1"]["percent"],
            "privacy_improvement_mia": privacy_improvement,
        },
    }

    # Save results
    results_file = Path("privacy_shield_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    logger.info("\n" + "=" * 80)
    logger.info("PRIVACY SHIELD DEMONSTRATION COMPLETE")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    try:
        results = demo_privacy_shield()
        print("\n[SUCCESS] Privacy Shield demonstration completed successfully!")
        print(f"\nKey Results:")
        print(f"  Fraud Detection F1:    Baseline={results['baseline']['metrics']['f1']:.4f} -> Protected={results['protected']['metrics']['f1']:.4f}")
        print(f"  MIA Attack Accuracy:   Baseline={results['baseline']['mia_attack']['attack_accuracy']:.4f} -> Protected={results['protected']['mia_attack']['attack_accuracy']:.4f}")
        print(f"  Privacy Improvement:   {results['comparison']['privacy_improvement_mia']:+.2f}%")

    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)
        sys.exit(1)
