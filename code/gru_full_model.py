import wandb
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from keras.models import Sequential, Model
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2, l1, l2
from keras.utils import to_categorical
import tensorflow as tf
from create_data_splits import create_data_splits # TODO: create_data_splits_pca
from get_metrics import get_metrics
from itertools import product

# Select Modalities
def modalities_combination_data_prep(modalities_combination_vec, part_fusion, df):
    selected_modalities_df = df.iloc[:, :3] #CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('selected_modalities_df', selected_modalities_df.columns)

    modalities_separated_df_list = []

    if part_fusion != 'early':
        if modalities_combination_vec[0]: # audio
            selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 59:123]], axis=1)
            selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 177:241]], axis=1)
        if modalities_combination_vec[1]: # face
            selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 10:58]], axis=1)
            selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 128:176]], axis=1)
        if modalities_combination_vec[2]: # talk
            selected_modalities_df = pd.concat([selected_modalities_df, df['s1']], axis=1)
            selected_modalities_df = pd.concat([selected_modalities_df, df[['s4','s5']]], axis=1)
            selected_modalities_df = pd.concat([selected_modalities_df, df[['s122','s123']]], axis=1)
        modalities_separated_df_list.append(selected_modalities_df)

    else:
        if modalities_combination_vec[0]: # audio
            selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 59:123]], axis=1)
            selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 177:241]], axis=1)
        if modalities_combination_vec[1]: # face
            selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 10:58]], axis=1)
            selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 128:176]], axis=1)
        if modalities_combination_vec[2]: # talk
            selected_modalities_df = pd.concat([selected_modalities_df, df['s1']], axis=1)
            selected_modalities_df = pd.concat([selected_modalities_df, df[['s4','s5']]], axis=1)
            selected_modalities_df = pd.concat([selected_modalities_df, df[['s122','s123']]], axis=1)
        #now, select only colums that have '_p1' on top, plus first 3 cols

        selected_modalities_df_p1 =  df.iloc[:, :3]
        columns_p1 = [col for col in selected_modalities_df.columns if '_p0' in col]
        df_cols_p1 = selected_modalities_df[columns_p1]
        selected_modalities_df_p1 = pd.concat([selected_modalities_df_p1,df_cols_p1], axis=1)
        modalities_separated_df_list.append(selected_modalities_df_p1)
        columns_p2 = [col for col in selected_modalities_df.columns if col not in columns_p1] #CHECK IF THIS IS RIGHT FOR SHMMER
        selected_modalities_df_p2 = selected_modalities_df[columns_p2]
        modalities_separated_df_list.append(selected_modalities_df_p2)


        
    return modalities_separated_df_list

def modalities_separation_data_prep(modalities_combination_vec,part_fusion, df):
    selected_modalities_df = df.iloc[:, :3]

    modalities_separated_df_list = []
    if modalities_combination_vec[0]: # audio
        selected_modalities_audio_df = pd.concat([selected_modalities_df, df.iloc[:, 59:123]], axis=1)
        selected_modalities_audio_df = pd.concat([selected_modalities_audio_df, df.iloc[:, 177:241]], axis=1)
        modalities_separated_df_list.append(selected_modalities_audio_df)
    if modalities_combination_vec[1]: # face
        selected_modalities_face_df = pd.concat([selected_modalities_df, df.iloc[:, 10:58]], axis=1)
        selected_modalities_face_df = pd.concat([selected_modalities_face_df, df.iloc[:, 128:176]], axis=1)
        modalities_separated_df_list.append(selected_modalities_face_df)
    if modalities_combination_vec[2]: # talk
        selected_modalities_talk_df = pd.concat([selected_modalities_df, df['s1']], axis=1)
        selected_modalities_talk_df = pd.concat([selected_modalities_talk_df, df[['s4','s5']]], axis=1)
        selected_modalities_talk_df = pd.concat([selected_modalities_talk_df, df[['s122','s123']]], axis=1)
        modalities_separated_df_list.append(selected_modalities_talk_df)

    if part_fusion != 'early':
        new_modalities_separated_df_list = []
        for item in modalities_separated_df_list:
            df_p1 = item.iloc[:,:3]
            columns_p1 = [col for col in item.columns if '_p1' in col]
            df_cols_p1 = item[columns_p1]
            df_p1 = pd.concat([df_p1,df_cols_p1], axis=1)
            new_modalities_separated_df_list.append(df_p1)
            columns_p2 = [col for col in item.columns if col not in columns_p1]
            df_p2 = item[columns_p2]
            new_modalities_separated_df_list.append(df_p2)

        modalities_separated_df_list = new_modalities_separated_df_list
    
    return modalities_separated_df_list

def build_gru_model(sequence_length, input_shape, num_gru_layers, gru_units, activation, use_bidirectional, dropout, reg):
    model = Sequential()
    model.add(Input(shape=(sequence_length, input_shape)))

    if num_gru_layers == 1:
        if use_bidirectional:
            model.add(Bidirectional(GRU(gru_units, activation=activation, kernel_regularizer=reg)))
        else:
            model.add(GRU(gru_units, activation=activation, kernel_regularizer=reg))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
    else:
        for _ in range(num_gru_layers - 1):
            if use_bidirectional:
                model.add(Bidirectional(GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)))
            else:
                model.add(GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg))
            model.add(Dropout(dropout))
            model.add(BatchNormalization())

        if use_bidirectional:
            model.add(Bidirectional(GRU(gru_units, activation=activation)))
        else:
            model.add(GRU(gru_units, activation=activation))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

    return model

def train():

    wandb.init()
    config = wandb.config
    print(config)
    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

    num_gru_layers = config.num_gru_layers
    gru_units = config.gru_units
    batch_size = config.batch_size
    epochs = config.epochs
    activation = config.activation_function
    use_bidirectional = config.use_bidirectional
    dropout = config.dropout_rate
    optimizer = config.optimizer
    learning_rate = config.learning_rate
    dense_units = config.dense_units
    kernel_regularizer = config.recurrent_regularizer
    loss = config.loss
    sequence_length = config.sequence_length
    part_fusion = config.part_fusion
    groundtruth = config.groundtruth

    test_metrics_list = {
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_accuracy_tolerant": [],
        "test_precision_tolerant": [],
        "test_recall_tolerant": [],
        "test_f1_tolerant": []
    }
    
    if (config.feature_set_tag == 'Stat'):
        if config.groundtruth == 'multi':
            stat_feature_df = pd.read_csv("../data/sign_features_05.csv")
            stat_feature = stat_feature_df['feature'].tolist()
        else:
            stat_feature_df = pd.read_csv("../data/sign_features_sign05.csv")
            stat_feature = stat_feature_df['feature'].tolist()
    elif (config.feature_set_tag == 'RF'):
        if config.groundtruth == 'multi':
            #skip this run
            return None
            
        else:
            rf_feature_df = pd.read_csv("../data/rf_top_sign05.csv")
            rf_feature = rf_feature_df['feature'].tolist()
            
    for fold in range(5):
        # select dataset and modalities
        if (config.dataset == 'clean'):
            if config.groundtruth == 'multi':
                df = pd.read_csv("../data/all_data_05.csv")
            else:
                df = pd.read_csv("../data/all_data_05_sign.csv")
        elif (config.dataset == 'normalized'):
            if config.groundtruth == 'multi':
                df = pd.read_csv("../data/all_data_05_norm.csv")
            else:
                df = pd.read_csv("../data/all_data_05_norm_sign.csv")
        else:
            if config.groundtruth == 'multi':
                df = pd.read_csv("../data/all_data_05_pca.csv")
            else:
                df = pd.read_csv("../data/all_data_05_pca_sign.csv")
        if (config.fusion_type == 'early') and (config.dataset != 'pca'):
            selected_modalities_list = modalities_combination_data_prep(config.modalities_combination, part_fusion, df)
            print('selected_modalities_list[0]', selected_modalities_list[0].columns)
            X_train_sequences_list = []
            X_val_sequences_list = []
            X_test_sequences_list = []
            feature_inputs = []
            for selected_modalities_df in selected_modalities_list:
                if (config.feature_set_tag == 'Stat'): # filter out feature not in stat list
                    repeat_columns = ['session','timeelapsed','groundtruth']
                    repeat_columns += [f for f in stat_feature if f in selected_modalities_df.columns]
                    if len(repeat_columns) <= 3: # no repeated feature
                        continue
                    selected_modalities_df = selected_modalities_df[repeat_columns]
                                        
                elif (config.feature_set_tag == 'RF'): # filter out feature not in rf list
                    repeat_columns = ['session','timeelapsed','groundtruth']
                    repeat_columns += [f for f in rf_feature if f in selected_modalities_df.columns]
                    if len(repeat_columns) <= 3: # no repeated feature
                        continue
                    selected_modalities_df = selected_modalities_df[repeat_columns]
                    
                splits = create_data_splits(
                    selected_modalities_df,
                    fold_no=fold,
                    num_folds=5,
                    seed_value=42,
                    sequence_length=sequence_length
                )
                
                X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits
                
                X_train_sequences_list.append(X_train_sequences)
                X_val_sequences_list.append(X_val_sequences)
                X_test_sequences_list.append(X_test_sequences)
                feature_inputs.append(Input(shape=(sequence_length, X_train_sequences.shape[2])))
            
            if loss == "categorical_crossentropy":
                y_train_sequences = to_categorical(y_train_sequences)
                y_test_sequences = to_categorical(y_test_sequences)
                y_val_sequences = to_categorical(y_val_sequences)
                
            if kernel_regularizer == "l1":
                reg = l1(0.01)
            elif kernel_regularizer == "l2":
                reg = l2(0.01)
            elif kernel_regularizer == "l1_l2":
                reg = l1_l2(0.01, 0.01)
            else:
                reg = None

            if part_fusion == 'early':
                input_shape = X_train_sequences.shape[2]
                model = build_gru_model(sequence_length, input_shape, num_gru_layers, gru_units, activation, use_bidirectional, dropout, reg)
                if loss == "categorical_crossentropy":
                    num_classes = len(np.unique(y_train_sequences))
                    model.add(Dense(dense_units, activation=activation))
                    model.add(Dense(num_classes, activation="softmax"))
                else:
                    model.add(Dense(dense_units, activation=activation))
                    model.add(Dense(1, activation="sigmoid"))

            else: 
                feature_outputs = []

                for feature_input in feature_inputs:
                    x = feature_input
                    for _ in range(num_gru_layers):
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                        x = Dropout(dropout)(x)
                        x = BatchNormalization()(x)
                    feature_outputs.append(x)

                concatenated_features = concatenate(feature_outputs)

                x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(concatenated_features)
                x = Dropout(dropout)(x)
                x = BatchNormalization()(x)
                
                if loss == "categorical_crossentropy":
                    y_train_sequences = to_categorical(y_train_sequences)
                    y_test_sequences = to_categorical(y_test_sequences)
                    y_val_sequences = to_categorical(y_val_sequences)
                    num_classes = len(np.unique(y_train_sequences))
                    x = Dense(dense_units, activation=activation)(x)
                    x = Dense(num_classes, activation="softmax")(x)
                else:
                    x = Dense(dense_units, activation=activation)(x)
                    x = Dense(1, activation="sigmoid")(x)

                model = Model(inputs=feature_inputs, outputs=x)
                
            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            elif optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
            elif optimizer == 'adadelta':
                optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
            
            model.summary()
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

            model_history = model.fit(
                X_train_sequences_list, y_train_sequences,
                validation_data=(X_val_sequences_list, y_val_sequences),
                epochs=epochs,
                batch_size=batch_size,
                verbose=2
            )
            y_predict_probs = model.predict(X_test_sequences_list)
            
        elif (config.fusion_type == 'intermediate') and (config.dataset != 'pca'):
            modalities_separated_df_list = modalities_separation_data_prep(config.modalities_combination, part_fusion, df)                
            X_train_sequences_list = []
            X_val_sequences_list = []
            X_test_sequences_list = []
            feature_inputs = []
            for selected_modalities_df in modalities_separated_df_list:
                if (config.feature_set_tag == 'Stat'): # filter out feature not in stat list
                    repeat_columns = ['session','timeelapsed','groundtruth']
                    repeat_columns += [f for f in stat_feature if f in selected_modalities_df.columns]
                    if len(repeat_columns) <= 3: # no repeated feature
                        continue
                    selected_modalities_df = selected_modalities_df[repeat_columns]
                                        
                elif (config.feature_set_tag == 'RF'): # filter out feature not in rf list
                    repeat_columns = ['session','timeelapsed','groundtruth']
                    repeat_columns += [f for f in rf_feature if f in selected_modalities_df.columns]
                    if len(repeat_columns) <= 3: # no repeated feature
                        continue
                    selected_modalities_df = selected_modalities_df[repeat_columns]
                    
                splits = create_data_splits(
                    selected_modalities_df,
                    fold_no=fold,
                    num_folds=5,
                    seed_value=42,
                    sequence_length=sequence_length
                )
                
                X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits
                
                X_train_sequences_list.append(X_train_sequences)
                X_val_sequences_list.append(X_val_sequences)
                X_test_sequences_list.append(X_test_sequences)
                feature_inputs.append(Input(shape=(sequence_length, X_train_sequences.shape[2])))
            
            if len(feature_inputs) < 1: # empty feature
                return None
            
            if kernel_regularizer == "l1":
                reg = l1(0.01)
            elif kernel_regularizer == "l2":
                reg = l2(0.01)
            elif kernel_regularizer == "l1_l2":
                reg = l1_l2(0.01, 0.01)
            else:
                reg = None 
                
            feature_outputs = []
            for feature_input in feature_inputs:
                x = feature_input
                for _ in range(num_gru_layers):
                    x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                    x = Dropout(dropout)(x)
                    x = BatchNormalization()(x)
                feature_outputs.append(x)

            if part_fusion == 'late':
                #concatenate every odd and even element, which corresponds to combinining the modalities for a single participant (assuming 2 participants)
                concatenated_features_list = []
                p1_features = feature_outputs[0::2]
                p2_features = feature_outputs[1::2]
                for p1, p2 in zip(p1_features, p2_features):
                    concatenated_features = concatenate([p1, p2])
                    concatenated_features_list.append(concatenated_features)
                

                second_feature_outputs = []
                for feature_output in concatenated_features_list:
                    x = feature_output
                    for _ in range(num_gru_layers):
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                        x = Dropout(dropout)(x)
                        x = BatchNormalization()(x)
                    second_feature_outputs.append(x)

                concatenated_features = concatenate(second_feature_outputs)
            else:
                concatenated_features = concatenate(feature_outputs)

            x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(concatenated_features)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
                
            if loss == "categorical_crossentropy":
                y_train_sequences = to_categorical(y_train_sequences)
                y_test_sequences = to_categorical(y_test_sequences)
                y_val_sequences = to_categorical(y_val_sequences)
                num_classes = len(np.unique(y_train_sequences))
                x = Dense(dense_units, activation=activation)(x)
                x = Dense(num_classes, activation="softmax")(x)
            else:
                x = Dense(dense_units, activation=activation)(x)
                x = Dense(1, activation="sigmoid")(x)

            model = Model(inputs=feature_inputs, outputs=x)

            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            elif optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
            elif optimizer == 'adadelta':
                optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
            model.summary()
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
            
            model_history = model.fit(
                X_train_sequences_list, y_train_sequences,
                validation_data=(X_val_sequences_list, y_val_sequences),
                epochs=epochs,
                batch_size=batch_size,
                verbose=2
            ) 
            y_predict_probs = model.predict(X_test_sequences_list)

        elif (config.fusion_type=='late') and (config.dataset != 'pca'): # fusion_type == late
            modalities_separated_df_list = modalities_separation_data_prep(config.modalities_combination, part_fusion, df)
            feature_inputs = []
            feature_outputs = []
            X_train_sequences_list = []
            X_val_sequences_list = []
            X_test_sequences_list = []
            for selected_modalities_df in modalities_separated_df_list:
                if (config.feature_set_tag == 'Stat'): # filter out feature not in stat list
                    repeat_columns = ['session','timeelapsed','groundtruth']
                    repeat_columns += [f for f in stat_feature if f in selected_modalities_df.columns]
                    if len(repeat_columns) <= 3: # no repeated feature
                        continue
                    selected_modalities_df = selected_modalities_df[repeat_columns]
                                                            
                elif (config.feature_set_tag == 'RF'): # filter out feature not in rf list
                    repeat_columns = ['session','timeelapsed','groundtruth']
                    repeat_columns += [f for f in rf_feature if f in selected_modalities_df.columns]
                    if len(repeat_columns) <= 3: # no repeated feature
                        continue
                    selected_modalities_df = selected_modalities_df[repeat_columns]
                    
                splits = create_data_splits(
                    selected_modalities_df,
                    fold_no=fold,
                    num_folds=5,
                    seed_value=42,
                    sequence_length=sequence_length
                )
                
                X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits
                
                X_train_sequences_list.append(X_train_sequences)
                X_val_sequences_list.append(X_val_sequences)
                X_test_sequences_list.append(X_test_sequences)
                feature_inputs.append(Input(shape=(sequence_length, X_train_sequences.shape[2])))

            if kernel_regularizer == "l1":
                reg = l1(0.01)
            elif kernel_regularizer == "l2":
                reg = l2(0.01)
            elif kernel_regularizer == "l1_l2":
                reg = l1_l2(0.01, 0.01)
            else:
                reg = None 
                
                
            if part_fusion == 'intermediate':

                for feature_input in feature_inputs:
                    
                    x = feature_input
                    for _ in range(num_gru_layers):
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                        x = Dropout(dropout)(x)
                        x = BatchNormalization()(x)
                    feature_outputs.append(x)

                #concatenate every 2 feature outputs, which corresponds to combinining the participants for a single modality (assuming 2 participants)
                concatenated_features_list = []
                for i in range(0, len(feature_outputs), 2):
                    concatenated_features = concatenate([feature_outputs[i], feature_outputs[i+1]])
                    concatenated_features_list.append(concatenated_features)
                
                second_feature_outputs = []
                for feature_output in concatenated_features_list:
                    x = feature_output
                    for _ in range(num_gru_layers):
                        x = GRU(gru_units, return_sequences=True, activation=activation, kernel_regularizer=reg)(x)
                        x = Dropout(dropout)(x)
                        x = BatchNormalization()(x)
                    second_feature_outputs.append(x)

                concatenated = concatenate(second_feature_outputs)

            
            elif part_fusion == 'late':
                for feature_input in feature_inputs:
                    feature_input = Input(shape=(sequence_length, X_train_sequences.shape[2]))
                    feature_model = build_gru_model(sequence_length, X_train_sequences.shape[2], num_gru_layers - 1, gru_units, activation, use_bidirectional, dropout, kernel_regularizer)
                    feature_output = feature_model(feature_input)
                    feature_outputs.append(feature_output)

                concatenated = concatenate(feature_outputs)

            
            
            if len(feature_inputs) < 1: # empty feature
                return None

            x = GRU(gru_units, activation=activation, kernel_regularizer=reg)(concatenated)
            x = Dropout(dropout)(x)
            x = BatchNormalization()(x)
            
            if loss == "categorical_crossentropy":
                y_train_sequences = to_categorical(y_train_sequences)
                y_test_sequences = to_categorical(y_test_sequences)
                y_val_sequences = to_categorical(y_val_sequences)
                num_classes = len(np.unique(y_train_sequences))
                x = Dense(dense_units, activation=activation)(x)
                output = Dense(num_classes, activation="softmax")(x)
            else:
                x = Dense(dense_units, activation=activation)(x)
                output = Dense(1, activation="sigmoid")(x)
        
            model = Model(inputs=feature_inputs, outputs=output)
            
            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            elif optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
            elif optimizer == 'adadelta':
                optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
            model.summary()
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
            
            model_history = model.fit(
                X_train_sequences_list, y_train_sequences,
                validation_data=(X_val_sequences_list, y_val_sequences),
                epochs=epochs,
                batch_size=batch_size,
                verbose=2
            )
                
            y_predict_probs = model.predict(X_test_sequences_list)

                
        else: # pca, skip modalities selection, skip fusion # TODO: do pca after modalities selection
            selected_modalities_df = df

            splits = create_data_splits(
                selected_modalities_df,
                fold_no=fold,
                num_folds=5,
                seed_value=42,
                sequence_length=sequence_length
            )
            
            # X_train_sequences, y_train_sequences => balanced sequences
            X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits
                
            print("X_train_sequences shape:", X_train_sequences.shape)
            print("X_test_sequences shape:", X_test_sequences.shape)
            
            if loss == "categorical_crossentropy":
                y_train_sequences = to_categorical(y_train_sequences)
                y_test_sequences = to_categorical(y_test_sequences)
                y_val_sequences = to_categorical(y_val_sequences)
            
            if kernel_regularizer == "l1":
                reg = l1(0.01)
            elif kernel_regularizer == "l2":
                reg = l2(0.01)
            elif kernel_regularizer == "l1_l2":
                reg = l1_l2(0.01, 0.01)
            else:
                reg = None
            
            input_shape = X_train_sequences.shape[2]
            model = build_gru_model(sequence_length, input_shape, num_gru_layers, gru_units, activation, use_bidirectional, dropout, reg)
                            
            if loss == "categorical_crossentropy":
                num_classes = len(np.unique(y_train_sequences))
                model.add(Dense(dense_units, activation=activation))
                model.add(Dense(num_classes, activation="softmax"))
            else:
                model.add(Dense(dense_units, activation=activation))
                model.add(Dense(1, activation="sigmoid"))
                
            if optimizer == 'adam':
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            elif optimizer == 'sgd':
                optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
            elif optimizer == 'adadelta':
                optimizer = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
            model.summary()
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
            
            model_history = model.fit(
                X_train_sequences, y_train_sequences,
                validation_data=(X_val_sequences, y_val_sequences),
                epochs=epochs,
                batch_size=batch_size,
                verbose=2
            )
            y_predict_probs = model.predict(X_test_sequences)

        for epoch in range(len(model_history.history['loss'])):
            metrics = {
                f't{fold}_loss': model_history.history['loss'][epoch],
            }
            if 'accuracy' in model_history.history:
                metrics[f't{fold}_accuracy'] = model_history.history['accuracy'][epoch]
            if 'precision' in model_history.history:
                metrics[f't{fold}_precision'] = model_history.history['precision'][epoch]
            if 'recall' in model_history.history:
                metrics[f't{fold}_recall'] = model_history.history['recall'][epoch]
            if 'auc' in model_history.history:
                metrics[f't{fold}_auc'] = model_history.history['auc'][epoch]
            
            wandb.log(metrics)

        print('Y TEST SEQUENCES', y_test_sequences)
        print(y_predict_probs)
        
        if loss == "categorical_crossentropy":
            y_pred = np.argmax(y_predict_probs, axis=1)
            y_test_sequences = np.argmax(y_test_sequences, axis=1)
        else:
            y_pred = (y_predict_probs > 0.5).astype(int).flatten()
            y_test_sequences = y_test_sequences.astype(int).flatten()

        test_metrics = get_metrics(y_pred, y_test_sequences, tolerance=1)
        for key in test_metrics:
            test_metrics_list[key].append(test_metrics[key])
        
        #put "test" before metrics
        test_metrics = {f"t{fold}_{k}": v for k, v in test_metrics.items()}
        wandb.log(test_metrics)
        if config.groundtruth == 'sign':
            #get confusion matrix
            conf_mat = tf.math.confusion_matrix(y_test_sequences, y_pred).numpy()
            #log the matrix (numerical)
            wandb.log({f"t{fold}_conf_mat" : conf_mat})
            #log y_pred, y_test_sequences
            wandb.log({f"t{fold}_y_pred" : y_pred})
            wandb.log({f"t{fold}_y_test_sequences" : y_test_sequences})
            print(conf_mat)
            
        #wandb.log({f"t{fold}_conf_mat" : wandb.plot.confusion_matrix(probs=None,
        #        y_true=y_test_sequences, preds=y_pred,
        #        class_names=['neutral', 'is_discomfort'])})
        
        print(test_metrics)
        
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)
    

def main():
    
    # Generate all combinations of 4 modalities in Boolean
    modalities_combinations = list(product([True, False], repeat=3)) # ['audio', 'face', 'talk']
    modalities_combinations = [comb for comb in modalities_combinations if any(comb)] # remove all False combination
    
    sweep_config = {
        'method': 'random',
        'name': 'gru_tuning',
        'parameters': {
            'feature_set_tag': {'values': ['Full', 'Stat', 'RF']},  # Full, Stat, RF, Quali
            'dataset': {'values': ['clean', 'normalized','pca']},
            'use_bidirectional': {'values': [True, False]},
            'num_gru_layers': {'values': [1, 2, 3]},
            'gru_units': {'values': [64, 128, 256]},
            'dropout_rate': {'values': [0.0, 0.3, 0.5, 0.8]},
            'dense_units': {'values': [32, 64, 128,256]},
            'activation_function': {'values': ['tanh', 'relu', 'sigmoid']},
            'optimizer': {'values': ['adam', 'sgd', 'adadelta', 'rmsprop']},
            'learning_rate': {'values': [0.0001, 0.001, 0.005, 0.01]},
            'batch_size': {'values': [64, 128, 256]},
            'epochs': {'value': 200},
            'recurrent_regularizer': {'values': ['l1', 'l2', 'l1_l2']},
            'loss' : {'values' : ["binary_crossentropy", "categorical_crossentropy"]},
            'sequence_length' : {'values' : [2, 5, 10, 30, 60]},
            'fusion_type': {'values': ['early', 'intermediate', 'late']},
            'modalities_combination': {'values': modalities_combinations},
            'part_fusion': {'values': ['early','intermediate','late']},
            'groundtruth': {'values': ['multi', 'sign']}
        }
    }

    # print(sweep_config)

    sweep_id = wandb.sweep(sweep=sweep_config, project="interpersonal-gru")
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    main()