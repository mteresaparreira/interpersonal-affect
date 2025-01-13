import torch
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

'''
Generates and returns:
- an array of sequences of data
- an array of corresponding target values

Requires:
- an array of input data that will be used to create sequences 
- an array of target values corresponding to the data
- an array of ids or sessions that will be used to group sequences
- an integer equal to the length of each sequence that will be created.
'''
def create_sequences(data, target, sessions, sequence_length):
    sequences = []
    targets = []

    unique_sessions = np.unique(sessions)
    for session in unique_sessions:
        session_indices = np.where(sessions == session)[0]
        session_data = data[session_indices]
        session_target = target[session_indices]

        if len(session_data) >= sequence_length:
            for i in range(len(session_data) - sequence_length + 1):
                sequences.append(session_data[i : i + sequence_length])
                targets.append(session_target[i + sequence_length - 1])
    
    return np.array(sequences), np.array(targets)

'''
Requires:
- a dataframe consisting of features to be trained on and target values
- an integer equal to the number of folds to create for cross validation
- the index of the fold to be used for the current train validation test split
- an integer seed value for random number generator
- an integer equal to the length of each sequence that will be created.

Creates and returns:
- X_train: training set data
- y_train: training set targets
- X_val: validation set data
- y_val: validation set targets
- X_test: testing set data
- y_test: testing set targets
- X_train_sequences: sequences generated from training set data
- y_train_sequences: an array of corresponding target values
- sequence_length: the length of the sequences returned
'''
def create_data_splits(df, num_folds = 5, fold_no=0, seed_value=42, sequence_length=1):
    
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        features = df.iloc[:, 3:]
        target = df.iloc[:, 2].values.astype('int')
        sessions = df['session'].values
        
        fold_sessions = df['session'].unique()

        num_of_sessions = len(fold_sessions)

        train_size = int(np.floor(0.7 * num_of_sessions))
        val_size = int(np.ceil(0.2 * num_of_sessions))
        test_size = num_of_sessions - train_size - val_size

        np.random.shuffle(fold_sessions)
        
        train_folds = []
        val_folds = []
        test_folds = []

        for i in range(num_folds):
            start_train_index = i * val_size
            end_train_index = (start_train_index + train_size if start_train_index+train_size <= len(fold_sessions) else start_train_index +  train_size - len(fold_sessions))
            #print("start_train_index", start_train_index)   
            #print("end_train_index", end_train_index)

            if start_train_index >= end_train_index:
                #get index list that rolls over to the beginning of the list
                train_fold = np.concatenate((fold_sessions[start_train_index:], fold_sessions[:end_train_index]))
            else:
                train_fold = fold_sessions[start_train_index : end_train_index]

            val_train_index = end_train_index
            val_end_index = (val_train_index + val_size if val_train_index+val_size <= len(fold_sessions) else val_train_index +  val_size - len(fold_sessions))

            if val_train_index >= val_end_index:
                #get index list that rolls over to the beginning of the list
                val_fold = np.concatenate((fold_sessions[val_train_index:], fold_sessions[:val_end_index]))
            else:
                val_fold = fold_sessions[val_train_index : val_end_index]

            test_fold = np.setdiff1d(fold_sessions, np.concatenate((train_fold, val_fold)))


            train_folds.append(train_fold)
            val_folds.append(val_fold)
            test_folds.append(test_fold)


        train_fold = train_folds[fold_no]
        val_fold = val_folds[fold_no]
        test_fold = test_folds[fold_no]

        print("folds:", train_fold, val_fold, test_fold)

        train_indices = df[df['session'].isin(train_fold)].index
        val_indices = df[df['session'].isin(val_fold)].index
        test_indices = df[df['session'].isin(test_fold)].index

        X_train = features.loc[train_indices]
        y_train = target[train_indices]
        session_train = sessions[train_indices]
        print("train shapes", X_train.shape, y_train.shape)

        X_val = features.loc[val_indices]
        y_val = target[val_indices]
        session_val = sessions[val_indices]
        print("val shapes", X_val.shape, y_val.shape)

        X_test = features.loc[test_indices]
        y_test = target[test_indices]
        session_test = sessions[test_indices]
        print("test shapes", X_test.shape, y_test.shape)

        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train, sequence_length)
        X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, session_val, sequence_length) 
        X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, session_test, sequence_length)
        print("Train sequences shape:", X_train_sequences.shape, y_train_sequences.shape)

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_data_splits_ids(df, num_folds = 5, fold_no=0, seed_value=42, sequence_length=1):
    
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        features = df.iloc[:, 3:]
        target = df.iloc[:, 2].values.astype('int')
        sessions = df['session'].values
        
        fold_sessions = df['session'].unique()

        num_of_sessions = len(fold_sessions)

        train_size = int(np.floor(0.7 * num_of_sessions))
        val_size = int(np.ceil(0.2 * num_of_sessions))
        test_size = num_of_sessions - train_size - val_size

        print('num_of_sessions', num_of_sessions)   
        #print("train_size", train_size)
        #print("val_size", val_size)
        #print("test_size", test_size)

        np.random.shuffle(fold_sessions)
        
        train_folds = []
        val_folds = []
        test_folds = []

        for i in range(num_folds):
            start_train_index = i * val_size
            end_train_index = (start_train_index + train_size if start_train_index+train_size <= len(fold_sessions) else start_train_index +  train_size - len(fold_sessions))
            #print("start_train_index", start_train_index)   
            #print("end_train_index", end_train_index)

            if start_train_index >= end_train_index:
                #get index list that rolls over to the beginning of the list
                train_fold = np.concatenate((fold_sessions[start_train_index:], fold_sessions[:end_train_index]))
            else:
                train_fold = fold_sessions[start_train_index : end_train_index]

            val_train_index = end_train_index
            val_end_index = (val_train_index + val_size if val_train_index+val_size <= len(fold_sessions) else val_train_index +  val_size - len(fold_sessions))

            if val_train_index >= val_end_index:
                #get index list that rolls over to the beginning of the list
                val_fold = np.concatenate((fold_sessions[val_train_index:], fold_sessions[:val_end_index]))
            else:
                val_fold = fold_sessions[val_train_index : val_end_index]

            test_fold = np.setdiff1d(fold_sessions, np.concatenate((train_fold, val_fold)))


            train_folds.append(train_fold)
            val_folds.append(val_fold)
            test_folds.append(test_fold)


        return train_folds, val_folds, test_folds
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_data_splits_feats(df, num_folds = 5, fold_no=0, seed_value=42, sequence_length=1, feature_list=None, with_val = True):
    
    try:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

        features = df.iloc[:, 3:]
        print('features inside splits', features)

    

   
        target = df.iloc[:, 2].values.astype('int')
        sessions = df['session'].values
        
        fold_sessions = df['session'].unique()

        num_of_sessions = len(fold_sessions)

        if with_val:
            train_size = int(np.floor(0.7 * num_of_sessions))
            val_size = int(np.ceil(0.2 * num_of_sessions))
        else:
            train_size  = int(np.floor(0.9 * num_of_sessions))
            val_size = int(np.ceil(0.2 * num_of_sessions))
        test_size = num_of_sessions - train_size - val_size

        np.random.shuffle(fold_sessions)
        
        train_folds = []
        val_folds = []
        test_folds = []

        for i in range(num_folds):
            start_train_index = i * val_size
            end_train_index = (start_train_index + train_size if start_train_index+train_size <= len(fold_sessions) else start_train_index +  train_size - len(fold_sessions))
            #print("start_train_index", start_train_index)   
            #print("end_train_index", end_train_index)

            if start_train_index >= end_train_index:
                #get index list that rolls over to the beginning of the list
                train_fold = np.concatenate((fold_sessions[start_train_index:], fold_sessions[:end_train_index]))
            else:
                train_fold = fold_sessions[start_train_index : end_train_index]

            if with_val:
                val_train_index = end_train_index
                val_end_index = (val_train_index + val_size if val_train_index+val_size <= len(fold_sessions) else val_train_index +  val_size - len(fold_sessions))

                if val_train_index >= val_end_index:
                    #get index list that rolls over to the beginning of the list
                    val_fold = np.concatenate((fold_sessions[val_train_index:], fold_sessions[:val_end_index]))
                else:
                    val_fold = fold_sessions[val_train_index : val_end_index]
            else:
                val_fold = []

            test_fold = np.setdiff1d(fold_sessions, np.concatenate((train_fold, val_fold)))


            train_folds.append(train_fold)
            val_folds.append(val_fold)
            test_folds.append(test_fold)


        train_fold = train_folds[fold_no]
        if with_val:
            val_fold = val_folds[fold_no]
        else:
            val_fold = []
        test_fold = test_folds[fold_no]

        print("folds:", train_fold, val_fold, test_fold)

        train_indices = df[df['session'].isin(train_fold)].index
        if with_val:
            val_indices = df[df['session'].isin(val_fold)].index
        else:
            val_indices = []
        test_indices = df[df['session'].isin(test_fold)].index

        X_train = features.loc[train_indices]
        y_train = target[train_indices]
        session_train = sessions[train_indices]
        print("train shapes", X_train.shape, y_train.shape)

        if with_val:
            X_val = features.loc[val_indices]
            y_val = target[val_indices]
            session_val = sessions[val_indices]
            print("val shapes", X_val.shape, y_val.shape)
        else:
            X_val = None
            y_val = None
            session_val = None

        X_test = features.loc[test_indices]
        y_test = target[test_indices]
        session_test = sessions[test_indices]
        print("test shapes", X_test.shape, y_test.shape)

        X_train = X_train.reset_index(drop=True)
        if with_val:
            X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        X_train_sequences, y_train_sequences = create_sequences(X_train.values, y_train, session_train, sequence_length)
        if with_val:
            X_val_sequences, y_val_sequences = create_sequences(X_val.values, y_val, session_val, sequence_length) 
        else:
            X_val_sequences = None
            y_val_sequences = None
        X_test_sequences, y_test_sequences = create_sequences(X_test.values, y_test, session_test, sequence_length)
        print("Train sequences shape:", X_train_sequences.shape, y_train_sequences.shape)

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None