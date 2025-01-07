import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import wandb
from itertools import product
from get_metrics import get_metrics
from create_data_splits import create_data_splits_feats
import random

# Select Modalities
def modalities_combination_data_prep(modalities_combination_vec,  df):
    selected_modalities_df = df.iloc[:, :3] #CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #print('selected_modalities_df', selected_modalities_df.columns)


    if modalities_combination_vec[0]: # audio
        selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 59:123]], axis=1)
        #print('selected_modalities_df', selected_modalities_df.columns)
        selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 177:241]], axis=1)
    if modalities_combination_vec[1]: # face
        selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 10:58]], axis=1)        
        #print('selected_modalities_df', selected_modalities_df.columns)
        selected_modalities_df = pd.concat([selected_modalities_df, df.iloc[:, 128:176]], axis=1)
    if modalities_combination_vec[2]: # talk
        selected_modalities_df = pd.concat([selected_modalities_df, df['s1']], axis=1)
        #print('selected_modalities_df', selected_modalities_df.columns)
        selected_modalities_df = pd.concat([selected_modalities_df, df[['s4','s5']]], axis=1)
        #print('selected_modalities_df', selected_modalities_df.columns)
        selected_modalities_df = pd.concat([selected_modalities_df, df[['s122','s123']]], axis=1)

        
    return selected_modalities_df


def train():

    wandb.init()
    config = wandb.config
    print(config)
    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)

    
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
    
    fold_importances = []
    
    if (config.feature_set_tag == 'Stat'):
        if config.groundtruth == 'multi':
            stat_feature_df = pd.read_csv("../data/sign_features_05.csv")
        else:  
            stat_feature_df = pd.read_csv("../data/sign_features_sign05.csv")
        stat_feature = stat_feature_df['feature'].tolist()
        print('stat_feature', stat_feature)
    else:
        stat_feature = None

    
    
    for fold in range(5):
        
        if config.groundtruth == 'multi':
            if config.dataset == 'clean':
                df = pd.read_csv("../data/all_data_05.csv")
            else:
                df = pd.read_csv("../data/all_data_05_norm.csv")
        else:
            if config.dataset == 'clean':
                df = pd.read_csv("../data/all_data_05_sign.csv")
            else:
                df = pd.read_csv("../data/all_data_05_norm_sign.csv")
        
        selected_modalities_df = modalities_combination_data_prep(config.modalities_combination, df)
        #print('selected_modalities_list[0]', selected_modalities_list[0].columns)
        
        if (config.feature_set_tag == 'Stat'): # filter out feature not in stat list
            repeat_columns = ['session','timeelapsed','groundtruth']
            repeat_columns += [f for f in stat_feature if f in selected_modalities_df.columns]
            if len(repeat_columns) <= 3: # no repeated feature
                continue
            selected_modalities_df = selected_modalities_df[repeat_columns]

        #print('start splits')
                 
        splits = create_data_splits_feats(
            selected_modalities_df,
            fold_no=fold,
            num_folds=5,
            seed_value=42,
            sequence_length=1,
            feature_list = stat_feature,
            with_val = False
            )

        if splits is None:
            return

        X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences, sequence_length = splits
        
            

        #print('end splits')
        
        # balance training dataset
        if config.balanced:

            smote = SMOTE(random_state=42) 
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced = X_train
            y_train_balanced = y_train
        rf = RandomForestClassifier(
                random_state=seed_value,
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split
                )
                
            
        rf.fit(X_train, y_train)
    
        y_pred = rf.predict(X_test)
        test_metrics = get_metrics(y_pred, y_test, tolerance=1)
        keys_test = test_metrics.keys()
        print('Test Metrics:', test_metrics)
        #print('keys_test:', keys_test)
        for key in keys_test:
            test_metrics_list[key].append(test_metrics[key])
        test_metrics = {f"{fold}_{k}": v for k, v in test_metrics.items()}
        wandb.log(test_metrics)

        # wandb.log({f"t{fold}_conf_mat" : wandb.plot.confusion_matrix(probs=None,
        #         y_true=y_test.astype(int) , preds=y_pred.astype(int) ,
        #         class_names=['no_discomfort', 'is_discomfort'])})
        
        # print(test_metrics)
        print(confusion_matrix(y_test, y_pred))
        #print(f'Fold {fold} Feature Importance:{rf.feature_importances_}')
        fold_importances.append(rf.feature_importances_)

    feature_names = selected_modalities_df.columns.tolist()
    #print(feature_names,'THIS WAS NAMES')
    feature_names = feature_names[3:]
    #print(feature_names)
    # Calculate average metrics and log to wandb
    avg_test_metrics = {f"avg_{key}": np.mean(values) for key, values in test_metrics_list.items()}
    wandb.log(avg_test_metrics)

    print("Average Metrics Across Groups:", avg_test_metrics)

    #print("Fold Importances:", fold_importances)
    #make it numpy array
    fold_importances = np.array(fold_importances)
    #print("Fold Importances Shape:", fold_importances.shape)
    avg_feature_importances = np.mean(fold_importances, axis=0)
    print("Average Feature Importances:", avg_feature_importances)
    feature_importance_dict = {feature_names[i]: avg_feature_importances[i] for i in range(len(feature_names))}
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    print("Sorted Feature Importances:", sorted_feature_importance)
    #save as csv
    feature_importance_df = pd.DataFrame.from_dict(sorted_feature_importance, orient='index')
    feature_importance_df.to_csv(f"feature_importance_sign_{config.feature_set_tag}_{config.dataset}.csv")
    #plot a bar plot
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 10))
    plt.bar(sorted_feature_importance.keys(), sorted_feature_importance.values())
    plt.show()
    #save plot
    fig.savefig(f"feature_importance_sign_{config.feature_set_tag}_{config.dataset}.png")


    wandb.log({"feature_importances": feature_importance_dict})




def main():
   # Generate all combinations of 3 modalities in Boolean
    modalities_combinations = list(product([True, False], repeat=3)) # ['audio', 'face', 'talk']
    modalities_combinations = [comb for comb in modalities_combinations if any(comb)] # remove all False combination
    
    # Sweep configuration
    sweep_config = {
        'method': 'random',
        'name': 'random_forest_tuning',
        'parameters': {
            'feature_set_tag': {'values': ['Full']},#['Full', 'Stat']}, # Full, Stat, RF, Quali
            'dataset': {'values': ['clean']},#['clean', 'normalized']},
            'n_estimators': {'values': [250]},#[50,100,250,500,1000,2500]},
            'max_depth': {'values': [15]},#[5, 15,30,50,100]},
            'modalities_combination': {'values': [[True,True,True]]},#modalities_combinations},
            'min_samples_split': {'values':[10]},#[2, 5, 10, 15, 100]},
            'balanced': {'values': [True]},#[True, False]},
            'groundtruth': {'values': ['sign']}#['multi', 'sign']}
        }
    }

    #PARAMETERS FOR SIGN
    #'parameters': {
    #        'feature_set_tag': {'values': ['Full']},#['Full', 'Stat']}, # Full, Stat, RF, Quali
    #        'dataset': {'values': ['clean']},#['clean', 'normalized']},
    #        'n_estimators': {'values': [250]},#[50,100,250,500,1000,2500]},
    #        'max_depth': {'values': [15]},#[5, 15,30,50,100]},
    #        'modalities_combination': {'values': [[True,True,True]]},#modalities_combinations},
    #        'min_samples_split': {'values':[10]},#[2, 5, 10, 15, 100]},
    #        'balanced': {'values': [True]},#[True, False]},
    #        'groundtruth': {'values': ['sign']}#['multi', 'sign']}
    #    }
        
    print(sweep_config)
    
    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="interpersonal_rf")
    wandb.agent(sweep_id, function=train)



if __name__ == '__main__':
    main()