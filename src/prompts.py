def create_table2code_prompt(chart_type, 
                             table
                             ):

    prompt = f"Generate the matplotlib code that represents a {chart_type} of the tabular data below."
    prompt += "Provide only the code as output, including the table values represented as list or numpy arrays."
    prompt += '\n' + table
    return prompt


def create_qa_prompt(question, 
                     choices, 
                     answer_type='multiple choice', 
                     template='qwen2vl', 
                     cot=False,
                     warning_message=None,
                     image_input=True,
                     table_input=False,  
                     table='',
                     axis_input=False,
                     axis=''
                     ):
    if answer_type=='multiple choice':
        prompt = question + '\nProvide the correct answer among the following choices:\n' + '\n'.join(choices)
    elif answer_type =='free text':
        prompt = question
    elif answer_type=='rank':
        prompt = question + '\nProvide the answer as a Python list.\n'
    else:
        print('Invalid test')
        raise ValueError 
    
    #Add axis
    if axis_input and image_input:
        prompt += '\nBelow is a description of the charts axis labels or legend.\n' +axis
    #Add table
    if table_input and image_input:
        prompt += '\nBelow is a table containing the values represented in the chart.\n'  + table
    if table_input and not image_input:
        #Provide only the table as input
        prompt = table + '\n' +  prompt     


    #Warnings 
    if warning_message:
        prompt+= '\nBe careful, the following design flaw has been identified in the chart: ' + warning_message
    else:
        pass
    
    #CoT or Direct prompting
    if cot:
        prompt += "\nAnswer with detailed reasoning steps and provide your final answer as 'Answer: your_answer'. Let's think step by step.\n"
    else:

        prompt+= "\nProvide only the final answer to the question.\n"

    if template=='internvl2':
        prompt = '<image>\n' + prompt
        
         
    return prompt