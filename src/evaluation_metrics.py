from utils import *
import ast
import numpy as np
import math
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel

def evaluate_dataset(dataset, 
                     ground_truth='best_answer',
                     rounding=2):
    '''
    Compute evaluation metrics over the entire dataset
    '''

    prediction = [str(d['predicted_answer']) if type(d['predicted_answer'])!=list else d['predicted_answer'][0] for d in dataset]
    best_answer = [d[ground_truth] for d in dataset]
    answer_type = [d['answer_type'] for d in dataset]
    return round(100*evaluate_qa(prediction, best_answer, answer_type),rounding)

def get_EM(prediction, reference):
    '''
    Compute exact match between prediction and reference. If rereference is a list, loop through all items.
    '''
    prediction = post_process_prediction(prediction).lower().strip()
    if type(reference)==str:
        reference = reference.lower().strip()
        if prediction.replace('\n', '') == reference:
            return 1
        if is_substring(reference, prediction) and reference!='n/a':
            #Prediction is more verbose than the reference but contains the reference
            if not (reference=='no' and prediction=='cannot be inferred / inadequate information'):
                if not f"than {reference}" in prediction and not f"than the {reference}" in prediction and not f"{reference} and" in prediction and not f" and {reference}" in prediction and not f"{reference} then" in prediction and not f"then {reference}" in prediction and not f"compared to {reference}" in prediction:
                    return 1
        return 0
    else:
        #Reference is a list
        for ref in reference:
            ref = ref.lower().strip()
            if prediction == ref:
                return 1
            if is_substring(ref, prediction):
                if not (ref=='no' and prediction.lower()=='cannot be inferred / inadequate information'): 
                    if not f"than {ref}" in prediction and not f"{ref} and" in prediction and not f" and {ref}" in prediction and not f"{ref} then" in prediction and not f"then {ref}" in prediction and not f"compared to {ref}" in prediction:
                        return 1
        return 0
    


def compute_mcnemar_table(predictions1, predictions2):
    """
    Constructs the 2x2 contingency table for McNemar's test.
    """
    table = np.zeros((2, 2), dtype=int)
    
    for pred1, pred2 in zip(predictions1, predictions2):
        ground_truth = pred1["best_answer"] 

        correct1 = evaluate_qa([pred1["predicted_answer"]], [ground_truth], [pred1['answer_type']])
        correct2 = evaluate_qa([pred2["predicted_answer"]], [ground_truth], [pred2['answer_type']])
        
        if correct1 and correct2:
            table[0, 0] += 1  
        elif correct1 and not correct2:
            table[0, 1] += 1  
        elif not correct1 and correct2:
            table[1, 0] += 1  
        else:  # both incorrect
            table[1, 1] += 1  
    return table

def compute_mcnemmar_significance_score(predictions_model1, predictions_model2):
    contingency_table = compute_mcnemar_table(predictions_model1, predictions_model2)
    result = mcnemar(contingency_table, exact=True)
    print("McNemar test statistic:", result.statistic)
    print("p-value:", result.pvalue)
    return result.pvalue, result.statistic


def compute_paired_t_test_significance_score(non_misleading_values, misleading_values):
    '''
    Compute whether the Likert-scale ratings are significantly different given a non-misleading visualization or a misleading one.
    Args:
        non_misleading_values (list): ratings for all MLLMs for the non-misleading version of a given dataset pair
        misleading_values (list): ratings for all MLLMs for the misleading version of a given dataset pair
    '''
    stat, p_value = ttest_rel(non_misleading_values, misleading_values)
    print("Paired t-test statistic:", stat)
    print("p-value:", p_value)
    return p_value, stat


def evaluate_qa(predictions, references, answer_types, num_threshold=0.05):
    #Provide a list with a single item to get score per instance
    scores = []
    for p, r, a in zip(predictions, references, answer_types):
        if a=='multiple choice':
            #compute Exact Match
            scores.append(get_EM(p,r))
        elif a=='free text':
            if is_float(r):
                float_r = convert_to_float(r)
                try:
                    float_p = convert_to_float(p)
                    if abs(float_r-float_p)<= (num_threshold * float_r):
                        scores.append(1)
                    else:
                        scores.append(0)
                except:
                    scores.append(get_EM(p,r))
            else:   
                scores.append(get_EM(p,r))
        elif a == 'rank':
            #Convert the reference to a list and check whether the items appear in order in the output
            r = ast.literal_eval(r)
            if type(r[0])!=list:
                if is_list_in_order(r, p):
                    scores.append(1)
                else:
                    scores.append(0)
            else:
                #There are two reference lists
                if is_list_in_order(r[0], p) or is_list_in_order(r[1], p):
                    scores.append(1)
                else:
                    scores.append(0)
        else:
            raise ValueError('Invalid answer type')
    accuracy = sum(scores)/len(scores)
    return accuracy

def compute_random_baseline(split='misleading'):
    calvi_misleading = load_json('datasets/calvi_data.json')[:45]
    calvi_standard = load_json('datasets/calvi_data.json')[45:]
    chartom = load_json('datasets/chartom_data.json')[::2]
    real_world = load_json('datasets/real_world_data.json')
    vlat = load_json('datasets/vlat_data.json')
    if split=='misleading':
        dataset = calvi_misleading + chartom + real_world
    else:
        dataset = calvi_standard + chartom + vlat
    random_score = 0
    for d in dataset:
        if d['answer_type'] == 'multiple choice':
            random_score += 1/len(d['choices'])
        elif d['answer_type'] == 'rank':
            size = len(ast.literal_eval(d['best_answer']))
            random_score += 1/math.factorial(size)
        else:
            pass
    
    random_score /= len(dataset)
    return random_score