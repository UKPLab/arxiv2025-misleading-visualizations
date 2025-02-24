from tqdm import tqdm
from prompts import *
from utils import *
from llm_inference import *
from loaders import *
import transformers
import argparse
import random
import os


transformers.set_seed(42)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='calvi-chartom-real_world-vlat',  help="Datasets, separated by -")
    parser.add_argument('--output_folder', type=str, required=True,  help="The folder to store the predictions json files")
    parser.add_argument('--model', type=str, required=True,  help="Name of the model to run")
    parser.add_argument('--prompt_type', type=str, choices=['standard', 'warning_message'], default='standard', help="Type of prompt to use")
    parser.add_argument('--max_tokens', type=int, default=200,  help="Max number of generated tokens")
    parser.add_argument('--table_folder', type=str, default='',  help="Folder containing the table representations of the charts.")
    parser.add_argument('--axis_folder', type=str, default='',  help="Folder containing the axis information of the charts.")
    parser.add_argument('--image_input', type=int, default=1,  help="1 to use the image, set to 0 only if providing the table as an alternative.")
    parser.add_argument('--redrawn_charts_folder', type=str, default='',  help="Folder containing the new chart images.")
    parser.add_argument('--shuffle', type=int, default=0,  help="1 to shuffle the choices for multiple choice questions")
    parser.add_argument('--shuffle_seed', type=int, default=42,  help="Seed for randomly shuffling the choices for multiple choice questions")
    args = parser.parse_args()

    if args.shuffle:
        random.seed(args.shuffle_seed)

    m = args.model
    print(f'Generating answers for model {m}')

    #Load model
    if m=='GPT4V' or m=='GPT4o':
        model, tokenizer, image_processor, context_len= m, '', '', ''
    else:
        # pass
        model, tokenizer = load_model(m)
        image_processor, context_len = '', ''
        if m in ['tinychart/3B/', 'llava-chartinstruct/13B/']:
            tokenizer, image_processor, context_len = tokenizer
    print('Model loaded')

    dataset_file_paths = {'calvi': 'datasets/calvi_data.json',
                        'chartom': 'datasets/chartom_data.json',
                        'real_world': 'datasets/real_world_data.json',
                        'vlat': 'datasets/vlat_data.json',
                        'chartqa': 'datasets/chartqa_data.json',
                        }
    #Warning messages
    misleader_warnings = {'inverted axis': 'the y-axis displays values increasing from top to bottom or the x-axis displays values increasing from right to left.', 
                          'truncated axis': 'the y-axis starts at a value above zero, e.g., the axis starts at 5000 and ends at 10000.', 
                          'inappropriate axis range': 'the y-axis range is too broad or too narrow.', 
                          'inconsistent tick intervals': 'the tick values on one axis are not equally spaced, e.g., the ticks are 20, 30, 50.',
                          '3d': 'the chart includes three-dimensional effects.', 
                          'dual axis': 'there are two independent y-axis, one on the left and one on the right, with different scales.', 
                          'area encoding': 'values are encoded as areas instead of height, while a bar chart would be more appropriate.',
                          'inappropriate item order': 'the ticks on one axis are displayed in an unconvential order, e.g., the dates are not displayed in chronological order.', 
                          'inappropriate aggregation': 'the chart is aggregating data in an inappropriate way.',
                          'overplotting': 'the chart displays too many data points together in one place.',
                          'inappropriate use of pie chart': 'the data displayed in the pie chart does not sum to 100%.',
                          'misrepresentation': 'the data values are drawn disproportionately or not to scale'                
                 }
    
    misleaders = ['inverted axis', 'truncated axis', 'inappropriate axis range',
                  'inconsistent tick intervals', '3d', 'dual axis', 'area encoding',
                  'inappropriate item order', 'inappropriate aggregation', 'overplotting',
                  'inappropriate use of pie chart', 'misrepresentation']

    dataset_list = args.datasets.split('-')


    for i in range(len(dataset_list)):
        results = []    
        dataset = load_json(dataset_file_paths[dataset_list[i]])

        #Check for tabular data 
        tables_file = False     
        if args.table_folder!='': 
            if dataset_list[i] + '.json' in os.listdir('chart2table/'+args.table_folder):
                tables_file = load_json(os.path.join('chart2table/'+args.table_folder, dataset_list[i]+'.json')) 
        #Check for axis data
        axis_file = False     
        if args.axis_folder!='': 
            if dataset_list[i] + '.json' in os.listdir('chart2axis/'+args.axis_folder):
                axis_file = load_json(os.path.join('chart2axis/'+args.axis_folder, dataset_list[i]+'.json'))    
        #Check for new chart images
        redrawn_charts_file = False     
        if args.redrawn_charts_folder!='': 
            if dataset_list[i] + '.json' in os.listdir('table2code/'+args.redrawn_charts_folder):
                redrawn_charts_file= load_json(os.path.join('table2code/'+args.redrawn_charts_folder, dataset_list[i]+'.json'))   
            
        for d in tqdm(range(len(dataset))):
            warning_message = None
            #Check for custom warning
            if args.prompt_type=='warning_message':

                if dataset_list[i] in ['calvi', 'chartom', 'real_world']:
                    if dataset[d]['misleader'].lower() in misleaders:
                        warning_message = misleader_warnings[dataset[d]['misleader'].lower()]


            #Check for table:
            table = ''
            table_input = False
            if tables_file:
                table = tables_file[d]['conversion']
                table_input = True 

            #Check for axis input 
            axis = ''
            axis_input = False
            if axis_file:
                axis = axis_file[d]['conversion']
                axis_input = True            

            #Get image_path
            image_path = dataset[d]['image_path']
            #Replace the default chart by a redrawn chart without misleaders, if there is one available
            if redrawn_charts_file:
                if redrawn_charts_file[d]['new_image_path']!='' and redrawn_charts_file[d]['old_image_path']==image_path:
                    image_path = redrawn_charts_file[d]['new_image_path']
        
            if args.shuffle:
                random.shuffle(dataset[d]['choices'])
            prompt = create_qa_prompt(dataset[d]['question'],
                                    dataset[d]['choices'], 
                                    dataset[d]['answer_type'],
                                    template=m,
                                    warning_message= warning_message,
                                    image_input = args.image_input,
                                    table_input=table_input,
                                    table=table,
                                    axis_input=axis_input,
                                    axis=axis)
            if args.image_input:
                predicted_answer = generate_answer(image_path, prompt, tokenizer, image_processor, context_len, model, m, args.max_tokens)
            else:
                #The image is not provided as input
                predicted_answer = generate_answer(None, prompt, tokenizer, image_processor, context_len, model, m, args.max_tokens)

            if type(predicted_answer)==list:
                predicted_answer = predicted_answer[0]

            #Add predictions to the results
            r = {'image_path': image_path, 
                'question': dataset[d]['question'],
                'answer_type': dataset[d]['answer_type'],
                'chart_type': dataset[d]['chart_type'],
                'misleader': dataset[d]['misleader'],
                'best_answer': dataset[d]['best_answer'],
                'misleading_answer': 'N/A',
                'predicted_answer': predicted_answer
                }
            if dataset_list[i] in  ['calvi', 'real_world']:
                r['misleading_answer'] = dataset[d]['misleading_answer']
            results.append(r)

        os.makedirs('results_qa',exist_ok=True)
        os.makedirs(f'results_qa/{args.output_folder}',exist_ok=True)
        with open(f'results_qa/{args.output_folder}/{dataset_list[i]}.json', 'w') as json_file:
            json.dump(results, json_file, indent=4)






