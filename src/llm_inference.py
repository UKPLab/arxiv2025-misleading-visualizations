from openai import AzureOpenAI
import os
import numpy as np
import time
from google import genai
import mimetypes
from utils import *



SLEEP = 5

#Replace by your environment variables
try:
    gpt_client = AzureOpenAI(
                api_key=os.getenv("YOUR_ENVIRONMENT_VARIABLE_CONTAINING_YOUR_AZURE_API_KEY"),  
                api_version="2023-10-01-preview",
                azure_endpoint = os.getenv("YOUR_ENVIRONMENT_VARIABLE_CONTAINING_YOUR_AZURE_ENDPOINT")
                )
    gemini_client = genai.Client(api_key=os.getenv("YOUR_ENVIRONMENT_VARIABLE_CONTAINING_YOUR_GEMINI_API_KEY"))
except:
    pass


def generate_answer_llm(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    '''
    Generate answer with Qwen 2.5
    '''
    messages = [
                    {"role": "user", "content": prompt}  
                ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_tokens, do_sample=False)
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def generate_answer_qwen2vl(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    if image_path:
        image = Image.open(image_path)
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, {"type": "image"}, ],},]
    else: 
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, ],},]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, temperature=0, top_p=1)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]   
    return response


def generate_answer_internvl2(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    generation_config = dict(max_new_tokens=max_tokens, do_sample=False)
    pixel_values = load_image_internvl2(image_path, max_num=12).to(torch.bfloat16).cuda()
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response


def generate_answer_internvl2_38B(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    generation_config = dict(max_new_tokens=max_tokens, do_sample=False)
    pixel_values = load_image_internvl2(image_path, max_num=12).to(torch.float16).cuda()
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response


def generate_answer_llava(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    image = Image.open(image_path)
    conversation = [
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                        ],
                    },
                    ]
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = tokenizer(text, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    response = tokenizer.decode(output[0], skip_special_tokens=True).split('ASSISTANT: ')[1]
    return response


def generate_answer_ovis(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    image = Image.open(image_path).convert('RGB')
    query = f'<image>\n{prompt}'
    # format conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, tokenizer[0].pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=tokenizer[1].dtype, device=tokenizer[1].device)]
    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer[0].pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        response = tokenizer[0].decode(output_ids, skip_special_tokens=True)
        return response


def generate_answer_chartinstruction(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens):
    from llava_hr.eval.run_llava import eval_model as eval_llava_chartinstruct
    response = eval_llava_chartinstruct(image_path, prompt, model, tokenizer, image_processor, context_len, conv_mode="llava_v1", temperature=0, max_new_tokens=max_tokens)
    return response


def generate_answer_chartgemma(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    image = Image.open(image_path).convert('RGB')
    inputs = tokenizer(text=prompt, images=image, return_tensors="pt")
    prompt_length = inputs['input_ids'].shape[1]
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    generate_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
    response= tokenizer.batch_decode(generate_ids[:, prompt_length:],
                                     skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=False)[0]
    return response


def generate_answer_tinychart(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens):
    from tinychart.eval.run_tiny_chart import inference_model
    from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds
    output = inference_model([image_path], prompt, model, tokenizer, image_processor, context_len, conv_mode="phi", temperature=0, max_new_tokens=max_tokens)
    if 'Answer with detailed steps' in prompt:
        if 'Answer=' not in output:
            response = ''
        else:
            try:
                response = evaluate_cmds(parse_model_output(output))
                if isinstance(response, np.integer):
                    response = str(int(response))
            except:
                response = ''
    else:
        response = output
    return response


def generate_answer_gpt4v(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens):
    if model=='GPT4V':
        deployment_name='gpt4-vision'
    else:
        deployment_name='gpt-4o'
    content = [{"type": "text", "text": prompt}]
    if image_path:
        image64 = encode_image_gpt4(image_path)
        content += [{"type":"image_url","image_url":{"url":image64}}]
    messages=[
      { 
      "role": "user",
      "content": content,
    }
    ]

    try:
        completion = gpt_client.chat.completions.create(model=deployment_name, 
                                                    temperature=0,
                                                    messages=messages, 
                                                    max_tokens=max_tokens)
        output = completion.choices[0].message.content
    except Exception as e:
        print(e)
        output = e
    time.sleep(SLEEP)
    return output


def generate_answer_gemini(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens):
    #Prepare input
    if image_path:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        media_type, _ = mimetypes.guess_type(image_path)
        if media_type is None:                  
            media_type = "image/png"
    content = (genai.types.Part.from_bytes(
                data=img_bytes,
                mime_type=media_type,
                ), 
                prompt
            )
    #Generate response
    response = gemini_client.models.generate_content(contents= content, 
                                                     model= model,
                                              config={
                                                "temperature": 0.0,
                                                "top_p": 1,
                                                "top_k": 1,
                                                "max_output_tokens": max_tokens
                                            }
                                            )
    output = response.text
    usage = usage_input + usage_output
    time.sleep(SLEEP)
    return output


def generate_answer(image_path, prompt, tokenizer, image_processor, context_len, model, template, max_tokens=200):
    prompt_map = {
                  'internvl2.5': generate_answer_internvl2, 
                  'internvl2.5/38B/': generate_answer_internvl2_38B,
                  'llava-v1.6-vicuna': generate_answer_llava, 
                  'ovis1.6': generate_answer_ovis,
                  'qwen2vl': generate_answer_qwen2vl, 
                  'qwen2.5': generate_answer_llm,
                  #chart models
                  'chartinstruction': generate_answer_chartinstruction,
                  'chartgemma': generate_answer_chartgemma,
                  'tinychart': generate_answer_tinychart,
                  #Closed-source models
                  'gpt4V': generate_answer_gpt4v,
                  'gpt4o':generate_answer_gpt4v,
                  'gemini-1.5-flash': generate_answer_gemini,
                  'gemini-1.5-pro': generate_answer_gemini
                  }
    if template!='internvl2.5/38B/':
        generation_type = template.split('/')[0]
    else:
        generation_type = template
    answer = prompt_map[generation_type](image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens)
    return answer
