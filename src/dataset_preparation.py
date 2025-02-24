import pandas as pd
import os
import json
from utils import *


#script to prepare the data

if __name__=='__main__':
    #CALVI --> No preparation needed


    #CHARTOM
    #First step: Request the data to the authors of CHARTOM, download it, and place it in a folder named chartom
    #CHARTOM: A Visual Theory-of-Mind Benchmark for Multimodal Large Language Models
    #https://arxiv.org/abs/2408.14419

    #Load files
    question_files = os.listdir('chartom/FACT_questions')
    answers = pd.read_csv('chartom/FACT_answer_keys.csv').set_index('chart id')
    manipulations = pd.read_csv('chartom/chart_manipulations.csv').set_index('chart id')
    questions = {}
    for q in question_files:
        with open(f'chartom/FACT_questions/{q}','r') as file:
            questions[q[:-4]] = file.read()

    chartom = []
    for q in questions.keys():
        answer_type = answers.loc[q,'answer type']
        best_answer = answers.loc[q,'answer']
        misleader_id =  q.split('Q')[0] + '*_2'
        chart_type = manipulations.loc[misleader_id,'chart type']
        misleader = manipulations.loc[misleader_id,'planted manipulation']
        if answer_type in ['free text', 'rank']:
            choices = []
        else:
            choices = questions[q].split('/n')[0].split('/n')
            
        d = {
            'image_path':f'chartom/chart_images/{q}.png',
            'question': questions[q].split('/n')[0],
            'choices': choices,
            'best_answer': best_answer,
            'chart_type': chart_type,
            'misleader': misleader,
            'answer_type': answer_type
            }
        chartom.append(d)

    
    with open('datasets/chartom_data.json', "w") as file:
        json.dump(chartom, file, indent=4)


    #Real-world
    #Download images from https://github.com/leoyuholo/bad-vis-browser
    os.makedirs("real_world_images", exist_ok=True)
    real_world = load_json('datasets/real_world_data.json')
    for r in real_world:
        save_image(r['image_url'], r['image_path'])


    #VLAT
    vlat_questions = pd.read_csv('VLAT/VLAT Questions.csv')
    vlat_metadata = pd.read_csv('VLAT/VLAT Questions Metadata.csv').set_index('id')
    vlat = []
    for v in range(len(vlat_questions)):
        if vlat_questions.loc[v,'dropped']=='no':
            chart_type = vlat_metadata.loc[vlat_questions.loc[v,'id'],'vis']
            d = {
                'image_path': f'VLAT/Images/{vlat_questions.loc[v,"vis"]}.png',
                'question': vlat_questions.loc[v,"question"],
                'choices': vlat_questions.loc[v,"options"].split('; '),
                'best_answer': vlat_questions.loc[v,"correct"],
                'chart_type': chart_type,
                'misleader': 'N/A',
                'answer_type': 'multiple choice'
                }
            vlat.append(d)
    with open('vlat_data.json', "w") as file:
        json.dump(vlat, file, indent=4)
    
