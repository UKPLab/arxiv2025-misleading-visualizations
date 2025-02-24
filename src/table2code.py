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
    parser.add_argument('--table_folder', type=str, default='',  help="Folder containing the table representations of the charts.")
    parser.add_argument('--output_folder', type=str, required=True,  help="The folder to store the generated codes")
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

    
    #Create results directory
    os.makedirs('table2code', exist_ok=True)
    os.makedirs(os.path.join('table2code',args.output_folder), exist_ok=True)
    
    #Load model
    model, tokenizer = load_model(args.model)
    image_processor, context_len ='', ''
    

    dataset_list = args.datasets.split('-')
    for idx in range(len(dataset_list)):
        results = []
        dataset = load_json(dataset_file_paths[dataset_list[idx]])
        tables_file = load_json(os.path.join('chart2table/'+args.table_folder, dataset_list[idx]+'.json'))     
        os.makedirs(f'table2code/{args.output_folder}/{dataset_list[idx]}/', exist_ok=True)

        for d in tqdm(range(len(dataset))):


            table = tables_file[d]['conversion']

            chart_type = dataset[d]['chart_type'].lower()
                    
            if 'chart' not in chart_type and chart_type.lower() not in ['scatterplot', 'n/a'] :
                chart_type += ' chart'

            
            if chart_type in  ['bar chart', 'pie chart', 'line chart', 'area chart', 'bubble chart', 'histogram', 'scatterplot',
                               'stacked bar chart' ,'stacked area chart', 'n/a']:
                prompt = create_table2code_prompt(chart_type, table)

                image_path = None
                output = generate_answer(image_path, #No image provided
                                        prompt, 
                                        tokenizer, 
                                        image_processor,
                                        context_len,
                                        model, 
                                        args.model, 
                                        args.max_new_tokens
                                        )
                
                #Plot the chart and save it
                try:
                    #Remove plt.show() and other unnecessary content
                    output = output.replace('plt.show()','').replace('```','')
                    if output.split('\n')[0]=='python':
                        output = output[7:] 
                    if 'Note:' in output:
                        output = output.split('Note:')[0]
                    if 'python' in output:
                        output = output.split('python')[0]
                    
                    #Add function to save the figure
                    output+= f'\nplt.savefig("table2code/{args.output_folder}/{dataset_list[idx]}/{d}.png", dpi=300)\nplt.clf()\nplt.close()'
                    exec_globals =  {}
                    exec_locals = {}
                    exec(output, exec_globals, exec_locals)
                    results.append({
                        'new_image_path': f'table2code/{args.output_folder}/{dataset_list[idx]}/{d}.png',
                        'old_image_path': dataset[d]['image_path'],
                        'chart_type': dataset[d]['chart_type'],
                        'table': tables_file[d]['conversion'],
                        'code': output
                        })
                except Exception as e:
                    #Code did not compile
                    print(f'Error rendering {output}: {e}')

                    results.append({
                        'new_image_path': '',
                        'old_image_path': dataset[d]['image_path'],
                        'chart_type': dataset[d]['chart_type'],
                        'table': tables_file[d]['conversion'],
                        'code': output
                        })
            else:
                #Chart type is not covered
                results.append({
                    'new_image_path': '',
                    'old_image_path': dataset[d]['image_path'],
                    'chart_type': dataset[d]['chart_type'],
                    'table': tables_file[d]['conversion'],
                    'code': ''
                    })
        
        #Save results file for the dataset
        with open(f'table2code/{args.output_folder}/{dataset_list[idx]}.json', "w") as file:
            json.dump(results, file, indent=4)

