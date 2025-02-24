import os
import pandas as pd
from tqdm import tqdm
import argparse
from evaluation_metrics import *


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str, default='',  help="Folder containing the table representations of the charts.")
    parser.add_argument('--output_file', type=str, default='results_qa.csv',  help="File to save the evaluation scores.")
    #Iterate through all the files in the results folder
    args = parser.parse_args()

    root = args.result_folder
    data = []

    for o in tqdm(os.listdir(root)):
        results_folder = os.path.join(root,o)
        results_files =  os.listdir(results_folder)
        calvi = load_json(os.path.join(results_folder,'calvi.json'))
        calvi_misleading = calvi[:45]
        calvi_normal = calvi[45:]
        real_world_misleading = load_json(os.path.join(results_folder,'real_world.json'))
        chartom = load_json(os.path.join(results_folder,'chartom.json'))
        chartom_misleading = chartom[1::2]
        chartom_normal = chartom[::2]     
        vlat_normal = load_json(os.path.join(results_folder,'vlat.json'))
        misleading_all = calvi_misleading + real_world_misleading + chartom_misleading
        normal_all = calvi_normal + chartom_normal + vlat_normal
        #Get scores
        new_entry =  {'model':o}
        new_entry['misleading_all_accuracy'] = evaluate_dataset(misleading_all,'best_answer')
        new_entry['normal_all_accuracy'] = evaluate_dataset(normal_all,'best_answer')
        new_entry['calvi_normal_accuracy'] = evaluate_dataset(calvi_normal,'best_answer')
        new_entry['calvi_misleading_accuracy'] = evaluate_dataset(calvi_misleading,'best_answer')
        new_entry['chartom_normal_accuracy'] = evaluate_dataset(chartom_normal,'best_answer')
        new_entry['chartom_misleading_accuracy'] = evaluate_dataset(chartom_misleading,'best_answer')
        new_entry['real_world_misleading_accuracy'] = evaluate_dataset(real_world_misleading,'best_answer')
        new_entry['vlat_normal_accuracy'] = evaluate_dataset(vlat_normal,'best_answer')
        data.append(new_entry)
    
    pd.DataFrame(data).to_csv(args.output_file,index=False)

