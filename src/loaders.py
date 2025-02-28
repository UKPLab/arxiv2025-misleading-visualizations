from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
import torch
import math


def loader_llm_default(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, 
                                                 torch_dtype=torch.float16) 
    tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2.5/2B/': 24, 'InternVL2.5/4B/': 36, 'InternVL2.5/8B/': 32,
        'InternVL2.5/26B/': 48, 'InternVL2.5/38B/': 64}[model_name.split('--')[-1]]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def loader_internvl2(model_path):
    torch.cuda.empty_cache()
    device_map = split_model(model_path)
    if '38B' not in model_path:
        model = AutoModel.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16, 
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map=device_map).eval()
    else:
        #Load in 8 bit for 38B
        # Define the quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Use 8-bit precision
            llm_int8_threshold=6.0,  
            llm_int8_enable_fp32_cpu_offload=True  
            )
        model = AutoModel.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def loader_llava(model_path):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    tokenizer = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, 
                                                              torch_dtype=torch.float16, 
                                                              low_cpu_mem_usage=True) 
    model.to("cuda:0")
    return model, tokenizer


def loader_qwen2vl(model_path):
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, 
                                                        torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True,
                                                        # attn_implementation="flash_attention_2",
                                                        device_map="auto"
                                                        )#.to("cuda:0")
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    tokenizer = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    return model, tokenizer


def loader_ovis(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    return model, (text_tokenizer, visual_tokenizer)


def loader_chartinstruction(model_path):
    from llava_hr.model.builder import load_pretrained_model as load_pretrained_model_llava
    from llava_hr.mm_utils import get_model_name_from_path as get_model_name_llava
    tokenizer, model, image_processor, context_len = load_pretrained_model_llava(
                                                model_path, 
                                                model_base=None,
                                                model_name=get_model_name_llava(model_path),
                                                device="cuda:0"
                                                )

    return model, (tokenizer, image_processor,context_len)


def loader_chartgemma(model_path):
    from transformers import PaliGemmaForConditionalGeneration
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, 
                                                              torch_dtype=torch.float16, 
                                                              low_cpu_mem_usage=True)
    tokenizer = AutoProcessor.from_pretrained(model_path)
    model = model.to("cuda:0")
    return model, tokenizer


def loader_tinychart(model_path):
    from tinychart.model.builder import load_pretrained_model
    from tinychart.mm_utils import get_model_name_from_path
    tokenizer, model, image_processor, context_len = load_pretrained_model(
                                                model_path, 
                                                model_base=None,
                                                model_name=get_model_name_from_path(model_path),
                                                device="cuda:0" # device="cpu" if running on cpu
                                                )

    return model, (tokenizer, image_processor,context_len)


def load_model(model):
    '''
    Load a (M)LLM. Return the model and the tokenizer
    '''
    paths = {
    #By default, the models are loaded from HuggingFace. Replace by your own local paths in this dictionary if you want to load local models.
    #multimodal large language model
    'internvl2.5/2B/':'OpenGVLab/InternVL2_5-2B',
    'internvl2.5/4B/':'OpenGVLab/InternVL2_5-4B',
    'internvl2.5/8B/':'OpenGVLab/InternVL2_5-8B',
    'internvl2.5/26B/':'OpenGVLab/InternVL2_5-26B',
    'internvl2.5/38B/':'OpenGVLab/InternVL2_5-38B',
    'ovis1.6/9B/': 'AIDC-AI/Ovis1.6-Gemma2-9B',
    'ovis1.6/27B/': 'AIDC-AI/Ovis1.6-Gemma2-27B',
    'llava-v1.6-vicuna/7B/':'llava-hf/llava-v1.6-vicuna-7b-hf',
    'llava-v1.6-vicuna/13B/':'llava-hf/llava-v1.6-vicuna-13b-hf',
    'qwen2vl/2B/':'Qwen/Qwen2-VL-2B-Instruct',
    'qwen2vl/7B/':'Qwen/Qwen2-VL-7B-Instruct',
    #large language models
    'qwen2.5/7B/': 'Qwen/Qwen2.5-7B-Instruct',
    #Chart models
    'chartinstruction/13B/' : 'lewy666/llava-hr-ChartInstruction',
    'chartgemma/3B/': 'ahmed-masry/chartgemma',
    'tinychart/3B/': 'mPLUG/TinyChart-3B-768'
        }
    
    loader_map = {'internvl2.5': loader_internvl2,  
                  'llava-v1.6-vicuna': loader_llava, 
                  'ovis1.6': loader_ovis,
                  'qwen2vl':loader_qwen2vl, 'qwen2.5': loader_llm_default,
                  'chartinstruction': loader_chartinstruction,
                  'chartgemma': loader_chartgemma,
                  'tinychart': loader_tinychart
                 }
    model, tokenizer = loader_map[model.split('/')[0]](paths[model])
    return model, tokenizer
