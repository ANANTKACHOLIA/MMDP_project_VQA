import torch
import os
import pandas as pd

def load_statements(file):
    with open(file, 'r') as fp:
        statements = fp.read().splitlines()
        
    df = pd.DataFrame({"question": [], "answer": [], "image_id":[]})
    for i in range(0, len(statements), 2):
        img_id = __redundants_pattern.findall(statements[i])[0][3]
        question = statements[i].replace(__redundants_pattern.findall(statements[i])[0][0], "")
        record = {
            "question": question,
            "answer": statements[i+1],
            "image_id": img_id,
        }
        df = df.append(record, ignore_index=True)
    
    return df

def split_df(df, test_size=0.2, train_out=None, test_out=None):
    train_df, test_df = train_test_split(df, test_size=test_size)

    if train_out is not None:
        train_df.to_csv(train_out, index=False)
    
    if test_out is not None:
        test_df.to_csv(test_out, index=False)
        
    return train_df, test_df
