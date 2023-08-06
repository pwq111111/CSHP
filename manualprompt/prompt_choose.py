import openai
import re

openai.api_key = ''

def get_manual_prompts(DATASET):
    prompts_choose = []
    path_manual_prompts = '/media/pwq/project/CoOp/manualprompt/prompts_manual_' + DATASET + '.txt'
    with open(path_manual_prompts,'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts_choose.append((line[:-1]))

    with open('/media/pwq/project/CoOp/manualprompt/prompt.txt','r') as f:
        prompt = f.read()

    path_class_name = '/media/pwq/project/CoOp/DATA/' + DATASET + '/class_name.txt'
    class_name = []
    with open(path_class_name,'r') as f:
        lines = f.readlines()
        for line in lines:
            class_name.append((line[:-1]))

    prompts =prompt.format(class_name,prompts_choose)

    MODEL = "gpt-3.5-turbo-16k"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in linguistics."},
            {"role": "user", "content": prompts},
        ],
        temperature=0,
    )
    pattern = r'(["\'])(.*?)\1'
    print(prompts)
    return re.findall(pattern, response['choices'][0]['message']['content'])[0][1]

if __name__ == "__main__":
    print(get_manual_prompts('food-101'))
