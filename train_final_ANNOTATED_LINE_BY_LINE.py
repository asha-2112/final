# ============================================================================
# MARINE ENGINE AI - LINE-BY-LINE ANNOTATED VERSION
# This file preserves the original project code and adds explanations directly
# underneath the code. The explanations are comments, so they do not execute.
# The comments explain libraries, functions, arguments, calculations, outputs,
# and why each part exists in the overall LSTM pipeline.
# ============================================================================

import os
# Explanation: Imports Python's os module for paths, folders, files, and operating-system operations.
import json
# Explanation: Imports JSON support so the program can save pipeline settings in a readable .json file.
import shutil
# Explanation: Imports shutil for deleting generated folders and files when the pipeline is reset.
import numpy as np
# Explanation: Imports NumPy as np for arrays, numerical operations, NaN values, and converting sequence lists into tensors.
import pandas as pd
# Explanation: Imports pandas as pd for reading CSV data and creating/manipulating tables and DataFrames.
import tensorflow as tf
# Explanation: Imports TensorFlow, the deep-learning framework used to build, train, and run the LSTM model.
from sklearn.preprocessing import StandardScaler
# Explanation: Imports StandardScaler, which standardizes sensor values so different sensor scales are comparable.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, silhouette_score
# Explanation: Imports evaluation functions: classification_report gives class-wise metrics, confusion_matrix counts actual-vs-predicted classes, accuracy_score calculates overall accuracy, precision_recall_fscore_support calculates precision/recall/F1, and silhouette_score evaluates cluster separation.
from sklearn.cluster import KMeans
# Explanation: Imports KMeans clustering, used later to group similar LSTM representations into behavioural patterns.
from tensorflow.keras import Sequential, Model
# Explanation: Imports Sequential for building the LSTM layer-by-layer and Model for extracting the internal 32-dimensional embedding layer.
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
# Explanation: Imports the neural-network layers: Input defines the shape, LSTM learns time relationships, Dense performs learned transformations/classification, and Dropout helps reduce overfitting.
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Explanation: Imports training callbacks: EarlyStopping stops training when validation loss stops improving, while ReduceLROnPlateau lowers the learning rate when progress stalls.


BASE = os.path.dirname(os.path.abspath(__file__))
# Explanation: Gets the absolute folder containing this Python script; this makes all project paths independent of the current terminal location.
DATA = os.path.join(BASE, 'data', 'marine_engine_fault_dataset.csv')
# Explanation: Builds the full path to the input marine-engine fault CSV inside the project's data folder.
OUT = os.path.join(BASE, 'outputs')
# Explanation: Defines the folder for the main human-readable output files.
ANALYSIS = os.path.join(BASE, 'analysis_outputs')
# Explanation: Defines the folder for intermediate and reproducibility files.
MODEL_DIR = os.path.join(BASE, 'model')
# Explanation: Defines the folder where the trained Keras model will be saved.

SENSORS = [
# Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    'Shaft_RPM','Engine_Load','Fuel_Flow','Air_Pressure','Ambient_Temp',
    # Explanation: Adds this sensor column name to the list of 18 model input parameters.
    'Oil_Temp','Oil_Pressure','Vibration_X','Vibration_Y','Vibration_Z',
    # Explanation: Adds this sensor column name to the list of 18 model input parameters.
    'Cylinder1_Pressure','Cylinder1_Exhaust_Temp','Cylinder2_Pressure',
    # Explanation: Adds this sensor column name to the list of 18 model input parameters.
    'Cylinder2_Exhaust_Temp','Cylinder3_Pressure','Cylinder3_Exhaust_Temp',
    # Explanation: Adds this sensor column name to the list of 18 model input parameters.
    'Cylinder4_Pressure','Cylinder4_Exhaust_Temp'
    # Explanation: Defines text used by the surrounding output/configuration structure.
]
# Explanation: Closes the current Python list or function/layer definition.
LABEL_NAMES = ['Normal'] + [f'Fault {i}' for i in range(1, 8)]
# Explanation: Creates the human-readable class names: Normal plus Fault 1 through Fault 7.
SEQ_LENGTHS = [10, 20, 30]
# Explanation: Defines the three time-window lengths that will be experimentally compared.
RANDOM_STATE = 42
# Explanation: Sets a fixed random seed so operations that use randomness can be reproduced more consistently.
np.random.seed(RANDOM_STATE)
# Explanation: Applies the fixed seed to NumPy's random-number generator.
tf.random.set_seed(RANDOM_STATE)
# Explanation: Applies the fixed seed to TensorFlow's random-number generator.


def reset_generated_dirs():
# Explanation: Defines the function `reset_generated_dirs` so this group of operations can be called as a reusable step.
    os.makedirs(OUT, exist_ok=True)
    # Explanation: Creates the requested folder if it does not already exist; `exist_ok=True` prevents an error when it already exists.
    os.makedirs(ANALYSIS, exist_ok=True)
    # Explanation: Creates the requested folder if it does not already exist; `exist_ok=True` prevents an error when it already exists.
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Explanation: Creates the requested folder if it does not already exist; `exist_ok=True` prevents an error when it already exists.
    for folder in (OUT, ANALYSIS, MODEL_DIR):
    # Explanation: Loops through each generated-output folder so old files can be removed before the new run.
        for name in os.listdir(folder):
        # Explanation: Loops through every file or subfolder currently inside the selected output folder.
            path = os.path.join(folder, name)
            # Explanation: Builds the full path to the current file or subfolder.
            if os.path.isdir(path):
            # Explanation: Checks whether the current path is a directory.
                shutil.rmtree(path)
                # Explanation: Recursively deletes an old generated directory and everything inside it.
            else:
            # Explanation: Runs the alternative branch when the preceding condition is false.
                os.remove(path)
                # Explanation: Deletes an old generated file.


def make_sequences(X, y, ts, seq):
# Explanation: Defines the function `make_sequences` so this group of operations can be called as a reusable step.
    xs, ys, t = [], [], []
    # Explanation: Creates empty Python lists that will collect sequence inputs, target labels, and the timestamp at the end of each sequence.
    for i in range(seq - 1, len(X)):
    # Explanation: Slides a window through the time-series so every possible sequence of the requested length is created.
        xs.append(X[i-seq+1:i+1])
        # Explanation: Adds the current time-window of sensor values to the input-sequence list.
        ys.append(y[i])
        # Explanation: Stores the label at the final time step as the target for this sequence.
        t.append(ts[i])
        # Explanation: Stores the timestamp corresponding to the final time step of the sequence.
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=int), np.asarray(t)
    # Explanation: Converts the sequence lists into NumPy arrays with model-friendly numeric data types and returns them.


def build_model(seq_len):
# Explanation: Defines the function `build_model` so this group of operations can be called as a reusable step.
    model = Sequential([
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        Input(shape=(seq_len, len(SENSORS))),
        # Explanation: Defines the shape of one model input: sequence length by 18 sensor features.
        LSTM(64, return_sequences=False),
        # Explanation: Creates the LSTM layer with 64 memory units; `return_sequences=False` means only the final sequence representation is passed forward.
        Dropout(0.2),
        # Explanation: Randomly drops a fraction of connections during training to reduce overfitting.
        Dense(32, activation='relu', name='embedding'),
        # Explanation: Creates a 32-unit learned representation layer named `embedding`; these 32 features are later used for behavioural clustering.
        Dense(8, activation='softmax')
        # Explanation: Creates the final 8-class softmax layer, producing probabilities for Normal and Fault 1–7.
    ])
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    return model
    # Explanation: Returns the calculated result from this function to the code that called it.


def train_one(seq_len, Xs, y, ts, tr_end, va_end):
# Explanation: Defines the function `train_one` so this group of operations can be called as a reusable step.
    Xtr, ytr, _ = make_sequences(Xs[:tr_end], y[:tr_end], ts[:tr_end], seq_len)
    # Explanation: Creates training sequences only from the first 70% of the chronological data.
    Xva, yva, _ = make_sequences(Xs[tr_end:va_end], y[tr_end:va_end], ts[tr_end:va_end], seq_len)
    # Explanation: Creates validation sequences only from the next 15% of the chronological data.
    Xte, yte, tte = make_sequences(Xs[va_end:], y[va_end:], ts[va_end:], seq_len)
    # Explanation: Creates test sequences only from the final 15% of the chronological data.

    model = build_model(seq_len)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    callbacks = [
    # Explanation: Starts a list of Keras callbacks that control the training process.
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        # Explanation: Creates a callback that watches validation loss and stops training after seven non-improving epochs, restoring the best weights.
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
        # Explanation: Creates a callback that halves the learning rate when validation loss stops improving, but never lets it go below 1e-5.
    ]
    # Explanation: Closes the current Python list or function/layer definition.
    history = model.fit(
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        Xtr, ytr,
        # Explanation: Creates training sequences only from the first 70% of the chronological data.
        validation_data=(Xva, yva),
        # Explanation: Provides the training arrays and validation arrays to the model so it can learn and be monitored on unseen validation data.
        epochs=40,
        # Explanation: Allows training for up to 40 passes through the training data, unless early stopping ends it sooner.
        batch_size=64,
        # Explanation: Processes 64 training sequences at a time before updating the model weights.
        verbose=2,
        # Explanation: Controls how much progress information Keras prints during training or prediction.
        callbacks=callbacks
        # Explanation: Passes the early-stopping and learning-rate callbacks into training.
    )
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    val_prob = model.predict(Xva, verbose=0)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    val_pred = val_prob.argmax(axis=1)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    _, _, val_f1, _ = precision_recall_fscore_support(
    # Explanation: Extracts only the macro F1 value from the precision/recall/F1 function; the other returned values are intentionally ignored.
        yva, val_pred, labels=list(range(8)), average='macro', zero_division=0
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    )
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    test_prob = model.predict(Xte, verbose=0)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    test_pred = test_prob.argmax(axis=1)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    test_acc = accuracy_score(yte, test_pred)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    _, _, test_f1, _ = precision_recall_fscore_support(
    # Explanation: Extracts the test macro F1 value while ignoring the detailed per-class arrays at this stage.
        yte, test_pred, labels=list(range(8)), average='macro', zero_division=0
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    )
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    return model, history, val_f1, test_acc, test_f1, Xte, yte, tte, test_prob
    # Explanation: Returns the calculated result from this function to the code that called it.


def sensor_profile(fault_df, normal_df, top_n=5):
# Explanation: Defines the function `sensor_profile` so this group of operations can be called as a reusable step.
    rows = []
    # Explanation: Creates an empty list to collect sensor comparison results.
    for sensor in SENSORS:
    # Explanation: Repeats the sensor comparison for all 18 engine parameters.
        normal_median = float(normal_df[sensor].median())
        # Explanation: Calculates the median value of the current sensor under Normal test conditions.
        fault_median = float(fault_df[sensor].median())
        # Explanation: Calculates the median value of the current sensor for the fault samples being analysed.
        pct = ((fault_median - normal_median) / normal_median * 100) if normal_median != 0 else np.nan
        # Explanation: Calculates percentage change of the fault median relative to the normal median; if the normal median is zero, it records NaN instead of dividing by zero.
        rows.append((sensor, normal_median, fault_median, pct))
        # Explanation: Adds the current sensor's normal median, fault median, and percentage change to the results list.
    rows.sort(key=lambda x: abs(x[3]) if np.isfinite(x[3]) else -1, reverse=True)
    # Explanation: Sorts sensors by the absolute size of their percentage change so the strongest differences appear first.
    return rows[:top_n]
    # Explanation: Returns the calculated result from this function to the code that called it.



def generate_fault_class_reference(patterns, contrib_df, selected_seq, acc, mp, mr, mf):
# Explanation: Defines the function `generate_fault_class_reference` so this group of operations can be called as a reusable step.
    """Generate one clean CSV reference for all fault-class information."""
    # Explanation: This docstring documents the purpose of the function: creating one clean CSV reference containing the fault-class information.
    rows = []
    # Explanation: Creates an empty list to collect sensor comparison results.

    # Fault class summary
    # Explanation: This source comment documents the purpose of the following code section.
    class_info = {
    # Explanation: Creates a dictionary containing the human-readable meaning and reporting status for Fault 0–7.
        0: ('Normal operating condition', 'Baseline / normal reference', 'Not applicable'),
        # Explanation: Defines the description, analysis type, and physical-classification wording for this dataset fault label.
        1: ('Dataset-defined Fault Class 1', '2 behavioural patterns identified', 'Physical classification requires mapping from the source dataset documentation'),
        # Explanation: Defines the description, analysis type, and physical-classification wording for this dataset fault label.
        2: ('Dataset-defined Fault Class 2', '3 behavioural patterns identified', 'Physical classification requires mapping from the source dataset documentation'),
        # Explanation: Defines the description, analysis type, and physical-classification wording for this dataset fault label.
        3: ('Dataset-defined Fault Class 3', '2 behavioural patterns identified', 'Physical classification requires mapping from the source dataset documentation'),
        # Explanation: Defines the description, analysis type, and physical-classification wording for this dataset fault label.
        4: ('Dataset-defined Fault Class 4', 'Contributing parameters analysed', 'Physical classification requires mapping from the source dataset documentation'),
        # Explanation: Defines the description, analysis type, and physical-classification wording for this dataset fault label.
        5: ('Dataset-defined Fault Class 5', 'Contributing parameters analysed', 'Physical classification requires mapping from the source dataset documentation'),
        # Explanation: Defines the description, analysis type, and physical-classification wording for this dataset fault label.
        6: ('Dataset-defined Fault Class 6', 'Contributing parameters analysed', 'Physical classification requires mapping from the source dataset documentation'),
        # Explanation: Defines the description, analysis type, and physical-classification wording for this dataset fault label.
        7: ('Dataset-defined Fault Class 7', 'Contributing parameters analysed', 'Physical classification requires mapping from the source dataset documentation'),
        # Explanation: Defines the description, analysis type, and physical-classification wording for this dataset fault label.
    }
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    for fault in range(8):
    # Explanation: Loops through all eight dataset classes, from Fault 0 through Fault 7.
        meaning, analysis, physical = class_info[fault]
        # Explanation: Unpacks the three pieces of reference information for the current fault class.
        rows.append({
        # Explanation: Adds the current sensor's normal median, fault median, and percentage change to the results list.
            'Section': 'Fault Class Summary', 'Fault': f'Fault {fault}',
            # Explanation: Sets the section label so the final CSV can distinguish class summaries, behavioural patterns, contributing parameters, and model summary.
            'Pattern_or_Rank': '', 'Samples': '', 'Parameter': '',
            # Explanation: Stores either the behavioural pattern number or the contributing-parameter rank.
            'Observed_Behaviour_or_Change': '', 'Interpretation': analysis,
            # Explanation: Stores the readable sensor change or behaviour associated with this record.
            'Physical_Classification_Status': physical
            # Explanation: Records that a physical failure name requires mapping from the original dataset documentation rather than being invented by the model.
        })
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # Behavioural patterns for Faults 1-3
    # Explanation: This source comment documents the purpose of the following code section.
    if patterns is not None and not patterns.empty:
    # Explanation: Checks that behavioural-pattern results exist before adding them to the reference.
        for _, r in patterns.iterrows():
        # Explanation: Loops through each discovered behavioural-pattern row.
            rows.append({
            # Explanation: Adds the current sensor's normal median, fault median, and percentage change to the results list.
                'Section': 'Behavioural Pattern',
                # Explanation: Sets the section label so the final CSV can distinguish class summaries, behavioural patterns, contributing parameters, and model summary.
                'Fault': f"Fault {int(r['Fault'])}",
                # Explanation: Stores the human-readable fault class name for this record.
                'Pattern_or_Rank': f"Pattern {int(r['Pattern'])}",
                # Explanation: Stores either the behavioural pattern number or the contributing-parameter rank.
                'Samples': int(r['Samples']),
                # Explanation: Stores the number of samples represented by this pattern when available.
                'Parameter': str(r['Top_Parameters']),
                # Explanation: Stores the relevant sensor or sequence-level parameter description.
                'Observed_Behaviour_or_Change': str(r['Pattern_Description']),
                # Explanation: Stores the readable sensor change or behaviour associated with this record.
                'Interpretation': str(r['Interpretation']),
                # Explanation: Stores a scientifically cautious interpretation that avoids claiming physical causality.
                'Physical_Classification_Status': 'Physical classification requires mapping from the source dataset documentation'
                # Explanation: Records that a physical failure name requires mapping from the original dataset documentation rather than being invented by the model.
            })
            # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # Contributing parameters for Faults 4-7
    # Explanation: This source comment documents the purpose of the following code section.
    if contrib_df is not None and not contrib_df.empty:
    # Explanation: Checks that contributing-parameter results exist before adding them to the reference.
        for _, r in contrib_df.iterrows():
        # Explanation: Loops through every contributing-parameter record for Faults 4–7.
            pct = float(r['Percent_Change_vs_Normal'])
            # Explanation: Calculates percentage change of the fault median relative to the normal median; if the normal median is zero, it records NaN instead of dividing by zero.
            direction = 'higher' if pct > 0 else 'lower' if pct < 0 else 'similar'
            # Explanation: Converts the sign of the percentage change into the words higher, lower, or similar.
            rows.append({
            # Explanation: Adds the current sensor's normal median, fault median, and percentage change to the results list.
                'Section': 'Contributing Parameter',
                # Explanation: Sets the section label so the final CSV can distinguish class summaries, behavioural patterns, contributing parameters, and model summary.
                'Fault': f"Fault {int(r['Fault'])}",
                # Explanation: Stores the human-readable fault class name for this record.
                'Pattern_or_Rank': f"Rank {int(r['Rank'])}",
                # Explanation: Stores either the behavioural pattern number or the contributing-parameter rank.
                'Samples': '',
                # Explanation: Stores the number of samples represented by this pattern when available.
                'Parameter': str(r['Sensor']),
                # Explanation: Stores the relevant sensor or sequence-level parameter description.
                'Observed_Behaviour_or_Change': f"{direction} ({pct:+.3f}% vs normal median)",
                # Explanation: Stores the readable sensor change or behaviour associated with this record.
                'Interpretation': 'Descriptive data-derived association; not a proven physical fault subtype or causal root cause.',
                # Explanation: Stores a scientifically cautious interpretation that avoids claiming physical causality.
                'Physical_Classification_Status': 'Physical classification requires mapping from the source dataset documentation'
                # Explanation: Records that a physical failure name requires mapping from the original dataset documentation rather than being invented by the model.
            })
            # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # Model summary row
    # Explanation: This source comment documents the purpose of the following code section.
    rows.append({
    # Explanation: Adds the current sensor's normal median, fault median, and percentage change to the results list.
        'Section': 'Model Summary', 'Fault': 'All classes',
        # Explanation: Sets the section label so the final CSV can distinguish class summaries, behavioural patterns, contributing parameters, and model summary.
        'Pattern_or_Rank': '', 'Samples': '', 'Parameter': f'Sequence length: {selected_seq} seconds',
        # Explanation: Stores either the behavioural pattern number or the contributing-parameter rank.
        'Observed_Behaviour_or_Change': f'Accuracy {acc:.6f}; Macro Precision {mp:.6f}; Macro Recall {mr:.6f}; Macro F1 {mf:.6f}',
        # Explanation: Stores the readable sensor change or behaviour associated with this record.
        'Interpretation': 'LSTM learns relationships between sensor sequences and supplied dataset labels.',
        # Explanation: Stores a scientifically cautious interpretation that avoids claiming physical causality.
        'Physical_Classification_Status': 'Source-documentation mapping required for physical nomenclature.'
        # Explanation: Records that a physical failure name requires mapping from the original dataset documentation rather than being invented by the model.
    })
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    csv_path = os.path.join(OUT, 'FAULT_CLASS_REFERENCE.csv')
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    print('Clean fault reference created:', csv_path)
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.


def main():
# Explanation: Defines the function `main` so this group of operations can be called as a reusable step.
    reset_generated_dirs()
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    df = pd.read_csv(DATA)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    if not os.path.exists(DATA):
    # Explanation: Checks this condition before deciding whether the following block should execute.
        raise FileNotFoundError(f'Dataset not found: {DATA}')
        # Explanation: Raises a clear error explaining that the required dataset file could not be found.
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    missing = [c for c in SENSORS + ['Fault_Label'] if c not in df.columns]
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    if missing:
    # Explanation: Stops execution if any required sensor or target column is missing from the dataset.
        raise ValueError(f'Missing required columns: {missing}')
        # Explanation: Raises a clear error listing which required columns are missing.
    df = df.sort_values('Timestamp').reset_index(drop=True)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    y = df['Fault_Label'].astype(int).to_numpy()
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    ts = df['Timestamp'].astype(str).to_numpy()
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    X = df[SENSORS].astype(float).to_numpy()
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    n = len(df)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    tr_end = int(n * 0.70)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    va_end = int(n * 0.85)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    scaler = StandardScaler().fit(X[:tr_end])
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    Xs = scaler.transform(X)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 1. Controlled sequence-length experiment: 10, 20, 30 sec
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    sequence_rows = []
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    candidates = {}
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    for seq_len in SEQ_LENGTHS:
    # Explanation: Runs the same training/evaluation experiment separately for 10, 20, and 30-second sequences.
        print('\n' + '=' * 72)
        # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
        print(f'SEQUENCE LENGTH = {seq_len} seconds')
        # Explanation: Prints which sequence length is currently being tested.
        print('=' * 72)
        # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
        model, history, val_f1, test_acc, test_f1, Xte, yte, tte, test_prob = train_one(
        # Explanation: Trains the candidate model for the current sequence length and receives its validation and test results.
            seq_len, Xs, y, ts, tr_end, va_end
            # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        )
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        sequence_rows.append({
        # Explanation: Adds the current sequence length and its validation/test metrics to the experiment summary.
            'Sequence_Length': seq_len,
            # Explanation: Records the sequence length used by this candidate.
            'Validation_Macro_F1': val_f1,
            # Explanation: Records validation macro F1, which is the main metric used to choose the final sequence length.
            'Test_Accuracy': test_acc,
            # Explanation: Records the candidate model's test accuracy for comparison/reporting.
            'Test_Macro_F1': test_f1,
            # Explanation: Records the candidate model's test macro F1 for comparison/reporting.
            'Epochs_Trained': len(history.history.get('loss', []))
            # Explanation: Records how many epochs actually ran, which may be less than 40 because of early stopping.
        })
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        candidates[seq_len] = (model, history, Xte, yte, tte, test_prob)
        # Explanation: Stores the candidate model and its outputs under the sequence length so the selected model can be retrieved later.
        print(f'Validation Macro F1: {val_f1:.6f}')
        # Explanation: Prints the validation macro F1 for the current sequence-length experiment.
        print(f'Test Accuracy: {test_acc:.6f}')
        # Explanation: Prints the test accuracy for the current sequence-length experiment.
        print(f'Test Macro F1: {test_f1:.6f}')
        # Explanation: Prints the test macro F1 for the current sequence-length experiment.

    seq_table = pd.DataFrame(sequence_rows)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    seq_table.to_csv(os.path.join(ANALYSIS, 'sequence_length_comparison.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    selected_seq = int(seq_table.loc[seq_table['Validation_Macro_F1'].idxmax(), 'Sequence_Length'])
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    model, history, Xte, yte, tte, test_prob = candidates[selected_seq]
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    pred = test_prob.argmax(axis=1)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    conf = test_prob.max(axis=1)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    print('\n' + '=' * 72)
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print(f'SELECTED SEQUENCE LENGTH: {selected_seq} seconds')
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print('=' * 72)
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 2. Save model + training information
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    model.save(os.path.join(MODEL_DIR, 'final_lstm_model.keras'))
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    pd.DataFrame(history.history).to_csv(os.path.join(ANALYSIS, 'training_history.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    with open(os.path.join(ANALYSIS, 'pipeline_config.json'), 'w', encoding='utf-8') as f:
    # Explanation: Opens the specified file for writing so the program can save a text or JSON report.
        json.dump({
        # Explanation: Serializes the pipeline settings dictionary into readable JSON with indentation.
            'sequence_lengths_tested': SEQ_LENGTHS,
            # Explanation: Defines text used by the surrounding output/configuration structure.
            'selected_sequence_length': selected_seq,
            # Explanation: Defines text used by the surrounding output/configuration structure.
            'sensors': SENSORS,
            # Explanation: Defines text used by the surrounding output/configuration structure.
            'classes': 8,
            # Explanation: Defines text used by the surrounding output/configuration structure.
            'manual_signatures': False,
            # Explanation: Defines text used by the surrounding output/configuration structure.
            'class_weights': False,
            # Explanation: Defines text used by the surrounding output/configuration structure.
            'split': 'chronological 70/15/15',
            # Explanation: Defines text used by the surrounding output/configuration structure.
            'random_state': RANDOM_STATE
            # Explanation: Defines text used by the surrounding output/configuration structure.
        }, f, indent=2)
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 3. Final classification results
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    acc = accuracy_score(yte, pred)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    mp, mr, mf, _ = precision_recall_fscore_support(
    # Explanation: Calculates the final macro precision, macro recall, and macro F1 across all eight classes.
        yte, pred, labels=list(range(8)), average='macro', zero_division=0
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    )
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    report = pd.DataFrame(classification_report(
    # Explanation: Creates a DataFrame from sklearn's detailed class-wise precision, recall, F1-score, and support report.
        yte, pred, labels=list(range(8)), target_names=LABEL_NAMES,
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        output_dict=True, zero_division=0
        # Explanation: Requests sklearn to return the classification report as a Python dictionary so it can be converted to a DataFrame.
    )).T
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    report.to_csv(os.path.join(OUT, 'classification_report.csv'))
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    cm = pd.DataFrame(
    # Explanation: Creates a labeled DataFrame from the 8×8 confusion matrix.
        confusion_matrix(yte, pred, labels=list(range(8))),
        # Explanation: Counts how many samples from each actual class were predicted as each possible class.
        index=[f'Actual_{i}' for i in range(8)],
        # Explanation: Controls whether pandas writes its default row-number index into the CSV; False keeps the output clean.
        columns=[f'Pred_{i}' for i in range(8)]
        # Explanation: Names the DataFrame columns so each learned feature can be identified.
    )
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    cm.to_csv(os.path.join(OUT, 'confusion_matrix.csv'))
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    pred_df = pd.DataFrame({
    # Explanation: Starts the table that links each test timestamp to its actual label, predicted label, and confidence.
        'Timestamp': tte,
        # Explanation: Adds the timestamp associated with each test sequence.
        'Actual_Label': yte,
        # Explanation: Adds the ground-truth label supplied by the dataset.
        'Predicted_Label': pred,
        # Explanation: Adds the class predicted by the LSTM.
        'Confidence': conf
        # Explanation: Adds the highest softmax probability for the predicted class.
    })
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    pred_df.to_csv(os.path.join(ANALYSIS, 'test_predictions.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 4. Extract learned 32-D LSTM representation
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # Keras 3 Sequential models may not expose .input until explicitly called.
    # Explanation: This source comment documents the purpose of the following code section.
    # The model is already built through Input(), but we explicitly call it once
    # Explanation: This source comment documents the purpose of the following code section.
    # to make the graph/node available across Keras/TensorFlow versions.
    # Explanation: This source comment documents the purpose of the following code section.
    _ = model(Xte[:1], training=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    embed = Model(inputs=model.inputs, outputs=model.get_layer('embedding').output)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    emb = embed.predict(Xte, verbose=0)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    emb_df = pd.DataFrame(emb, columns=[f'LSTM_Feature_{i+1}' for i in range(32)])
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    emb_df['Timestamp'] = tte
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    emb_df['Actual_Label'] = yte
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    emb_df['Predicted_Label'] = pred
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    emb_df.to_csv(os.path.join(ANALYSIS, 'lstm_embeddings.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 5. Validated pattern discovery: only Faults 1-3
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    raw_test = df.iloc[va_end:].copy().reset_index(drop=True)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    raw_test['Timestamp'] = raw_test['Timestamp'].astype(str)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    merged = emb_df.merge(raw_test, on=['Timestamp'], how='inner', suffixes=('', '_raw'))
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    # The raw dataset uses Fault_Label, while the embedding table uses Actual_Label.
    # Explanation: This source comment documents the purpose of the following code section.
    # Keep the LSTM/evaluation label as the canonical label for downstream analysis.
    # Explanation: This source comment documents the purpose of the following code section.
    if 'Fault_Label' in merged.columns:
    # Explanation: Checks which version of the fault-label column exists after merging the embedding and raw test tables.
        merged['Actual_Label'] = merged['Actual_Label'].astype(int)
        # Explanation: Ensures the canonical evaluation label is stored as an integer.
    elif 'Actual_Label_raw' in merged.columns:
    # Explanation: Provides a fallback for the alternative merged column name if pandas created it.
        merged['Actual_Label'] = merged['Actual_Label'].astype(int)
        # Explanation: Ensures the canonical evaluation label is stored as an integer.

    pattern_rows = []
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    assign_rows = []
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    sensor_rows = []
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    validated_k = {1: 2, 2: 3, 3: 2}
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    normal = merged[merged['Actual_Label'] == 0]
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    for fault, k in validated_k.items():
    # Explanation: Loops through Faults 1–3 and their predefined validated cluster counts.
        sub = merged[merged['Actual_Label'] == fault].copy()
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        if len(sub) < k:
        # Explanation: Checks that there are enough samples to create the requested number of clusters.
            continue
            # Explanation: Skips this fault if there are not enough samples for the requested analysis.
        Z = sub[[f'LSTM_Feature_{i+1}' for i in range(32)]].to_numpy()
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit(Z)
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        sil = silhouette_score(Z, km.labels_) if len(np.unique(km.labels_)) > 1 else np.nan
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        sub['Pattern'] = km.labels_ + 1
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

        for _, row in sub[['Timestamp', 'Actual_Label', 'Predicted_Label', 'Pattern']].iterrows():
        # Explanation: Records the timestamp, labels, and discovered pattern assignment for each clustered sample.
            assign_rows.append(row.to_dict())
            # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

        for cluster_id in range(1, k + 1):
        # Explanation: Processes each discovered behavioural cluster one at a time.
            cluster = sub[sub['Pattern'] == cluster_id]
            # Explanation: Selects only the samples belonging to the current behavioural pattern.
            top = sensor_profile(cluster, normal, top_n=5)
            # Explanation: Finds the five sensors with the largest absolute median differences from normal for this pattern.
            desc = '; '.join(
            # Explanation: Builds a readable sentence-like description of whether each top sensor is higher or lower than normal and by how much.
                f'{s} ({"higher" if pct > 0 else "lower" if pct < 0 else "similar"}, {pct:+.1f}%)'
                # Explanation: Formats each sensor's name, direction, and percentage change into readable text.
                for s, _, _, pct in top
                # Explanation: Starts a loop that repeats the following indented operations for each item in the selected collection.
            )
            # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
            pattern_rows.append({
            # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
                'Fault': fault,
                # Explanation: Stores the human-readable fault class name for this record.
                'Pattern': cluster_id,
                # Explanation: Defines text used by the surrounding output/configuration structure.
                'Samples': len(cluster),
                # Explanation: Stores the number of samples represented by this pattern when available.
                'Silhouette': sil,
                # Explanation: Defines text used by the surrounding output/configuration structure.
                'Top_Parameters': ', '.join(x[0] for x in top),
                # Explanation: Defines text used by the surrounding output/configuration structure.
                'Pattern_Description': desc,
                # Explanation: Defines text used by the surrounding output/configuration structure.
                'Interpretation': 'Descriptive LSTM-representation behavioural pattern; not a proven physical fault subtype or causal root cause.'
                # Explanation: Stores a scientifically cautious interpretation that avoids claiming physical causality.
            })
            # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
            for sensor in SENSORS:
            # Explanation: Repeats the sensor comparison for all 18 engine parameters.
                nm = float(normal[sensor].median())
                # Explanation: Reads the normal median for this sensor.
                cm = float(cluster[sensor].median())
                # Explanation: Reads the current pattern/cluster median for this sensor.
                pct = ((cm - nm) / nm * 100) if nm != 0 else np.nan
                # Explanation: Calculates percentage change of the fault median relative to the normal median; if the normal median is zero, it records NaN instead of dividing by zero.
                sensor_rows.append({
                # Explanation: Stores the detailed normal median, pattern median, and percentage change for this sensor.
                    'Fault': fault,
                    # Explanation: Stores the human-readable fault class name for this record.
                    'Pattern': cluster_id,
                    # Explanation: Defines text used by the surrounding output/configuration structure.
                    'Sensor': sensor,
                    # Explanation: Defines text used by the surrounding output/configuration structure.
                    'Normal_Median': nm,
                    # Explanation: Defines text used by the surrounding output/configuration structure.
                    'Pattern_Median': cm,
                    # Explanation: Defines text used by the surrounding output/configuration structure.
                    'Percent_Change_vs_Normal': pct
                    # Explanation: Defines text used by the surrounding output/configuration structure.
                })
                # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    patterns = pd.DataFrame(pattern_rows)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    patterns.to_csv(os.path.join(OUT, 'discovered_patterns.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    pd.DataFrame(assign_rows).to_csv(os.path.join(ANALYSIS, 'pattern_cluster_assignments.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    pd.DataFrame(sensor_rows).to_csv(os.path.join(ANALYSIS, 'pattern_sensor_statistics.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 6. Contributing-parameter profiles for Faults 4-7
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    contrib = []
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    for fault in range(4, 8):
    # Explanation: Loops through Faults 4, 5, 6, and 7 for contributing-parameter analysis.
        sub = merged[merged['Actual_Label'] == fault]
        # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
        top = sensor_profile(sub, normal, top_n=5)
        # Explanation: Finds the five sensors with the largest absolute median differences from normal for this pattern.
        for rank, (sensor, nm, fm, pct) in enumerate(top, 1):
        # Explanation: Numbers the top five sensor differences from rank 1 through rank 5.
            contrib.append({
            # Explanation: Adds one contributing-parameter record containing the fault, rank, sensor, medians, and percentage change.
                'Fault': fault,
                # Explanation: Stores the human-readable fault class name for this record.
                'Rank': rank,
                # Explanation: Defines text used by the surrounding output/configuration structure.
                'Sensor': sensor,
                # Explanation: Defines text used by the surrounding output/configuration structure.
                'Normal_Median': nm,
                # Explanation: Defines text used by the surrounding output/configuration structure.
                'Fault_Median': fm,
                # Explanation: Defines text used by the surrounding output/configuration structure.
                'Percent_Change_vs_Normal': pct
                # Explanation: Defines text used by the surrounding output/configuration structure.
            })
            # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    contrib_df = pd.DataFrame(contrib)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    contrib_df.to_csv(os.path.join(OUT, 'contributing_parameters.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    contrib_df.to_csv(os.path.join(ANALYSIS, 'fault4_7_contributing_parameters.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 7. Clean human-readable fault reference
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    generate_fault_class_reference(patterns, contrib_df, selected_seq, acc, mp, mr, mf)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 8. Timestamp-level predicted fault occurrences
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    occ = pred_df.copy()
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    occ['Fault'] = occ['Predicted_Label'].where(occ['Predicted_Label'] > 0, 0)
    # Explanation: Creates a simplified fault column where Normal is 0 and predicted Fault 1–7 remain their class numbers.
    occ['Group'] = (occ['Fault'] != occ['Fault'].shift()).cumsum()
    # Explanation: Creates a new group number whenever the predicted fault changes, which identifies contiguous runs.
    events = occ[occ['Fault'] > 0].groupby(['Fault', 'Group']).agg(
    # Explanation: Groups each contiguous non-Normal predicted run and summarizes its start time, end time, number of samples, and mean confidence.
        Start=('Timestamp', 'first'),
        # Explanation: Takes the first timestamp in each predicted-fault run as its start time.
        End=('Timestamp', 'last'),
        # Explanation: Takes the last timestamp in each predicted-fault run as its end time.
        Samples=('Timestamp', 'size'),
        # Explanation: Counts how many consecutive test predictions belong to that predicted-fault run.
        Mean_Confidence=('Confidence', 'mean')
        # Explanation: Calculates the average model confidence across the predicted-fault run.
    ).reset_index(drop=True)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
    events.to_csv(os.path.join(ANALYSIS, 'fault_occurrences.csv'), index=False)
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.

    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    # 9. Final research report
    # Explanation: This source comment documents the purpose of the following code section.
    # ------------------------------------------------------------
    # Explanation: This source comment documents the purpose of the following code section.
    with open(os.path.join(OUT, 'FINAL_RESEARCH_REPORT.txt'), 'w', encoding='utf-8') as f:
    # Explanation: Opens the specified file for writing so the program can save a text or JSON report.
        f.write('MARINE ENGINE AI - FINAL CLEAN LSTM RESEARCH REPORT\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('=' * 72 + '\n\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('MODEL\n-----\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Single 8-class LSTM using 18 raw sensor parameters.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write(f'Sequence lengths tested: {SEQ_LENGTHS}.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write(f'Selected sequence length: {selected_seq} seconds.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Manual fault signatures: NONE.\nClass weights: NONE.\n\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('FINAL TEST METRICS\n------------------\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write(f'Accuracy: {acc:.6f}\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write(f'Macro Precision: {mp:.6f}\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write(f'Macro Recall: {mr:.6f}\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write(f'Macro F1: {mf:.6f}\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write(f'Test samples: {len(yte)}\n\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('FAULT CLASS REFERENCE\n---------------------\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Fault 0: Normal operating condition.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Faults 1-7: dataset-defined fault classes.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Physical classification status: the supplied dataset uses categorical fault labels, while the corresponding physical failure nomenclature requires mapping from the source dataset documentation.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Therefore, this project reports data-derived behavioural descriptions instead of inventing physical names.\n\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('A detailed CSV fault reference is available in outputs/FAULT_CLASS_REFERENCE.csv.\n\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('PATTERN ANALYSIS\n----------------\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Faults 1-3: validated LSTM-representation behavioural clustering (K=2,3,2).\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Faults 4-7: descriptive contributing-parameter analysis only.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Patterns are exploratory behavioural associations, not proven physical fault subtypes.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Sensor associations do not establish causal root causes.\n\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('FAULT OCCURRENCES\n-----------------\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Occurrences are contiguous runs of the same predicted fault label in timestamp-level test predictions.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('They are not a physical-event benchmark because the dataset does not provide physical event IDs.\n\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('LIMITATIONS\n-----------\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Dataset-specific evaluation; external-engine validation is not included.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Overlapping temporal windows are not statistically independent.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.
        f.write('Sensor associations do not establish physical causality.\n')
        # Explanation: Writes the specified report text into the final human-readable research report.

    print('\n' + '=' * 72)
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print('FINAL CLEAN PIPELINE COMPLETED')
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print('=' * 72)
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print(f'Selected sequence: {selected_seq} seconds')
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print(f'Accuracy: {acc:.4f}')
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print(f'Macro Precision: {mp:.4f}')
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print(f'Macro Recall: {mr:.4f}')
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print(f'Macro F1: {mf:.4f}')
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print('Main outputs:', sorted(os.listdir(OUT)))
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print('Intermediate outputs:', sorted(os.listdir(ANALYSIS)))
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.
    print('Model:', os.path.join(MODEL_DIR, 'final_lstm_model.keras'))
    # Explanation: Prints this information to the terminal so the user can monitor the pipeline run.


if __name__ == '__main__':
# Explanation: Runs the main() function only when this file is executed directly, not when it is imported by another Python file.
    main()
    # Explanation: Explanation: This line performs the operation shown in the code and contributes to the surrounding pipeline step.
