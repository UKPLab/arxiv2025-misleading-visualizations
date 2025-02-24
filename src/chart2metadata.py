import argparse
import os 
import json
from tqdm import tqdm
from transformers import set_seed
from utils import *
from loaders import *
from prompts import *
from llm_inference import *

SEED = 42
set_seed(SEED)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='calvi-chartom-real_world-vlat',  help="Datasets, separated by -")
    parser.add_argument('--output_folder', type=str, required=True,  help="The folder to store the predictions json files")
    parser.add_argument('--modality', type=str, default='table', choices=['table', 'axis'], help="Modality to convert the chart to")
    parser.add_argument('--model', type=str, required=True, help="Model to use for chart conversion")
    parser.add_argument('--max_new_tokens', type=int, default=2048, help="Number of new tokens to generate")
    args = parser.parse_args()

    #Prepare data
    dataset_file_paths = {'calvi': 'datasets/calvi_data.json',
                        'chartom': 'datasets/chartom_data.json',
                        'real_world': 'datasets/real_world_data.json',
                        'vlat': 'datasets/vlat_data.json',
                        'chartqa': 'datasets/chartqa_data.json',
                        }


    model, tokenizer = load_model(args.model)
    image_processor, context_len = '', ''

    dataset_list = args.datasets.split('-')
    for i in range(len(dataset_list)):
        results = []    

        dataset = load_json(dataset_file_paths[dataset_list[i]])

        for d in tqdm(range(len(dataset))):


            chart_type = dataset[d]['chart_type'].lower()
            #Prompt
            if args.modality=='table':
                #Table prediction
                prompt = "Generate the underlying data table of the figure below. Change columns with |, change row by starting a new line. Provide only the table as output."
            else:
                #Axis prediction
                if chart_type ==  'map':
                    prompt = "What is the legend and its categories (with their colors) in this map? Answer only with the content of the legend and its categories (with their colors)."
                elif chart_type in  ['pie', 'pie chart']:
                    prompt = "What are the categories in this pie chart? Answer only with the categories."
                else:
                    prompt = "What are the axis labels and ticks of this chart? Answer only with the axis labels and the tick values, going from the bottom-left to the top-left corner of the chart for the y-axis and from the bottom-left to the bottom-right corner for the x-axis."
            if 'internvl2' in args.model:
                prompt = '<image>\n' + prompt

            output = generate_answer(dataset[d]['image_path'], 
                                    prompt, 
                                    tokenizer, 
                                    image_processor,
                                    context_len,
                                    model, 
                                    args.model, 
                                    args.max_new_tokens)
            results.append({
                'image_path': dataset[d]['image_path'],
                'conversion': output,
                'chart_type': chart_type
                })

        #Save results
        os.makedirs(f'chart2{args.modality}', exist_ok=True)
        os.makedirs(f'chart2{args.modality}/{args.output_folder}', exist_ok=True)
        with open(f'chart2{args.modality}/{args.output_folder}/{dataset_list[i]}.json', "w") as file:
            json.dump(results, file, indent=4)

