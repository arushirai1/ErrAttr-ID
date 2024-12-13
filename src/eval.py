MULTI_GPU_MODE = True


from llama_models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)

global_caption_prompt ="Caption the image with the most salient details"


global_hypothesis_system_prompt = '''Formulate a hypothesis on why the misclassified images might be failing. Focus on properties of the image or attributes of the object in your hypothesis given the context.'''

global_hypothesis_few_shot_example = [
    UserMessage(content=f'''Context: "The actual class is 'ant', however model incorrectly predicts 'burrito' 4 times\n The actual class is 'hotdog', however model incorrectly predicts 'west_highland_white_terrier' 4 times\n The actual class is 'mushroom', however model incorrectly predicts 'pig' 4 times\n The actual class is 'tank', however model incorrectly predicts 'parachute' 4 times\n The actual class is 'African_chameleon', however model incorrectly predicts 'shih_tzu' 3 times" Description: "A sample with an incorrect prediction has description: The image features a woman wearing a yellow shirt with a gun design on it. The shirt has a green and red color scheme, and the gun is prominently displayed on the front. The image is taken in a dimly lit room, making the colors appear muted and details of the gun design less distinct."
                    "A sample with an incorrect prediction has description: The image is a black and white drawing of a small dog with a bow tie. The drawing was photographed under low light, causing shadows to obscure some details of the bow tie."
                    "A sample with a correct prediction has description: The image features a cartoon-like drawing of a lizard with a sad expression. Despite being in low light, the high contrast in the drawing ensures that the lizard's details remain visible."
                    "A sample with a correct prediction has description: The image features a colorful, hand-woven blanket with a bird design. The bright and vibrant colors make the details of the bird clear, even under dim lighting." \nHypothesis:'''),
                CompletionMessage(
                    content="""The model is likely misclassifying images in poorly lit conditions because details of the object are obscured.""",
                    stop_reason=StopReason.end_of_turn,
                )
]


global_labeling_system_prompt ="Create a concise labeling prompt given a hypothesis"
global_labeling_few_shot = [UserMessage(content=f'Hypothesis: The model is likely misclassifying images in poorly lit conditions because details of the object are obscured.\nLabeling Prompt: '),
                CompletionMessage(
                    content="""For each image, label from two options whether the lighting condition is: \n[1] 'Bright' \n[2] 'Dim/Dark'.""",
                    stop_reason=StopReason.end_of_turn,
                ),
    UserMessage(content=f"Hypothesis: The model struggles to classify 'bird' images due to difficulty recognizing the bird when it is partially obstructed or in a non-frontal pose.\nLabeling Prompt: "),
    CompletionMessage(
        content="""For each image, label from two options the view of the bird: \n[1] Frontal \n[2] Non-frontal or Obstructed.""",
        stop_reason=StopReason.end_of_turn,
    ),
    ]

global_label_images_system_prompt =""


# In[149]:


global_prune_labeling_output_prompt = [
                SystemMessage(content="Given the reasoning, only generate the best match: gather_observations, experiment, conclusion"),
                # TODO: Add examples
                UserMessage(content=f'Reasoning: Based on the context, the selected action is:\n\n(1) gather_observations\n\nThe context provides information on classes with the highest errors but does not provide enough information to conclude the error slice (specific attributes of erronous images/objects) or form a hypothesis that requires experimentation. Therefore, the next step is to gather more evidence to analyze the error slice further.\nSelected Action:'),
                CompletionMessage(
                    content="""gather_observations""",
                    stop_reason=StopReason.end_of_turn,
                ),
                UserMessage(content=f"Reasoning: The next logical step after identifying classes with the highest errors is to gather more information about the specific errors, which would allow for a more informed hypothesis and potentially an experiment to test it.\nSelected Action:"),
                CompletionMessage(
                    content="""experiment""",
                    stop_reason=StopReason.end_of_turn,
                ),
        ] # this is for prune() in LLM brain


global_agent_prompt = """
Goal: Identify error-slice attributes (e.g., object or image properties). Based on the context, select ONE ACTION:
1. gather_observations: Collect more details about the object class if context lacks information or an experiment has failed.
2. experiment: Formulate and test a hypothesis if observations from gather_observations or context provide some basis for a hypothesis.
3. conclusion: Observations and error rates should indicate that the majority of the error is correlated with a particular attribute, state conclusion by summarizing the attribute and object class (make sure experiments are done).

Rules:
- Start with gather_observations if details are insufficient or if only the error rates by class are provided.
- Once there are detailed descriptions or observations, conduct an experiment.
- Conduct experiment before drawing conclusions.
- Briefly justify the action and the object class the action will be applied to based on the context.
- Select only one object class to focus on.
""" # this was only selecting gather_observations

# global_agent_prompt = """**Goal:** Determine error-slice attributes. Select ONE ACTION:
# 1. gather_observations: Collect more data about the object class.
# 2. experiment: Test a hypothesis.
# 3. conclusion: State the error-slice attribute and object class.

# **Rules:**
# - gather_observations: When initial information is insufficient.
# - Conduct experiments before drawing conclusions.
# - Justify your action and the object class.
# """

global_agent_few_shot_prompt = [
    UserMessage(content="""Context: "The top five classes with the highest errors are: [('gorilla', np.float64(0.1013215859030837)), ('pineapple', np.float64(0.03482587064676617)), ('lion', np.float64(0.02702702702702703)), ('wine_bottle', np.float64(0.026785714285714284)), ('barn', np.float64(0.02564102564102564))]"\nSelected Action: """),
    CompletionMessage(
                    content="""The context shows that class with the highest error is 'gorilla', since we do not have any observations, the action should be to 'gather_observations'""",
                    stop_reason=StopReason.end_of_turn,
                ),
    UserMessage(content="""Context: The top five classes with the highest errors are: [('lemon', np.float64(0.07386363636363637)), ('toy_poodle', np.float64(0.03333333333333333)), ('pembroke_welsh_corgi', np.float64(0.031746031746031744)), ('baboon', np.float64(0.03007518796992481)), ('scottish_terrier', np.float64(0.028846153846153848))]
Since we have error rates by class but no detailed descriptions or observations, the action should be to 'gather_observations' to collect more details about the object class 'lemon', which has the highest error rate. This will provide a basis for further analysis.
lambda row: row['gt'] == 'lemon'
["A 'lemon' sample with a incorrect prediction has description: The image features a yellow cartoon character, possibly Pac-Man, running with a big smile on its face. The character is wearing a red ring around its neck and is surrounded by graffiti, giving the scene a vibrant and lively atmosphere. The character appears to be enjoying its time, possibly racing or participating in a fun activity.", "A 'lemon' sample with a incorrect prediction has description: The image features a yellow cartoon character with a red nose and antlers, walking across a white background. The character appears to be a deer, and it is walking towards the viewer. The deer is also wearing a pair of yellow shoes. The overall scene is a playful and whimsical representation of a deer.","A 'lemon' sample with a correct prediction has description: The image features a yellow stuffed lemon sitting in a silver strainer, surrounded by several lemons. The stuffed lemon appears to be smiling, adding a playful touch to the scene. The lemons are placed both inside and outside the strainer, creating a visually appealing arrangement.", "A 'lemon' sample with a correct prediction has description: The image features a plate filled with various fruits, including bananas, oranges, and apples. The plate is placed on a table, and the fruits are arranged in a visually appealing manner. The oranges are scattered throughout the plate, with some placed closer to the center and others towards the edges. The bananas are positioned towards the left side of the plate, while the apples are located more towards the right side. The overall scene creates a sense of abundance and freshness."]\nSelected Action: """),
    CompletionMessage(
                    content="""There are more details now, it seems like for 'lemon', the presence of "cartoon characters," "vibrant paintings," "graffiti," "black and white dots," and "stylized representations" in the incorrectly classified "lemon" images suggests the model may be struggling to accurately classify images that exhibit a computer graphic." Now we can conduct an experiment and have a basis for a hypothesis.""",
                    stop_reason=StopReason.end_of_turn,
    ),
]

# '''The goal of this system is to find specific error-slice attributes such as properties of an object or image. Given context, select ONE ACTION from the following actions: gather_observations, experiment, or conclusion.
#                                         (1) gather_observations: given the context, gather more descriptions for a particular object class.
#                                         (2) experiment: given some context and evidence, you need to form a hypothesis and evaluate to confirm. generally do experiment before conclusion
#                                         (3) conclusion: if there is enough information in the context to conclude the error slice
#                                         '''


# In[151]:


slice_condition_prompt = [
    UserMessage(
        content='''Context: The top five classes with the highest errors are: [('goose', np.float64(0.18699186991869918)), ('Granny_Smith', np.float64(0.03773584905660377)), ('fly', np.float64(0.0375)), ('bell_pepper', np.float64(0.03676470588235294)), ('fire_engine', np.float64(0.03636363636363636))]\n Slice Condition Code: '''),
    CompletionMessage(
        content="""lambda row: row['gt'] == 'goose' """,
        stop_reason=StopReason.end_of_turn,
    ),
    # UserMessage(
    #     content='''Context: The top five classes with the highest errors are: [('goose', np.float64(0.18699186991869918)), ('Granny_Smith', np.float64(0.03773584905660377)), ('fly', np.float64(0.0375)), ('bell_pepper', np.float64(0.03676470588235294)), ('fire_engine', np.float64(0.03636363636363636))]\n Slice Condition Code: '''),
    # CompletionMessage(
    #     content="""lambda row: row['gt'] == 'goose' """,
    #     stop_reason=StopReason.end_of_turn,
    # ),
    # UserMessage(content=f'Context: {context}\n Slice Condition Code: '),
    # CompletionMessage(
    #     content="""lambda row: row['gt'] == 'goose' """,
    #     stop_reason=StopReason.end_of_turn,
    # ),
    # UserMessage(
    #     content='''Context:  The top five classes with the highest errors are: [('cannon', np.float64(0.3698630136986301)), ('shield', np.float64(0.36496350364963503)), ('lipstick', np.float64(0.34177215189873417)), ('gazelle', np.float64(0.3381294964028777)), ('baseball_player', np.float64(0.32941176470588235))] 
    #     'Based on the context, the selected action is:\n\n(1) gather_observations\n\nThe context provides information about the top five marginal errors, descriptions of samples with incorrect predictions, and descriptions of samples with correct predictions. However, there is no clear indication of what specific error-slice attribute (property of an object or image) is being sought. Therefore, gathering evidence is the most appropriate action to take next, as it would involve examining the descriptions and images further to identify any patterns or characteristics that may be contributing to the errors.',
    #     Previous slice_condition = lambda row: row['gt'] == 'cannon'
    #     "Results of investigating cannon experiment: {'**Cartoon/Abstract**': 0.5, '**Realistic/High-Res**': 0.5, '**Other**': 0, 'Unknown': 0.5}"
    #     There is not enough information to conclude, so we will gather evidence again except for a different class \n Slice Condition Code: '''),
    # CompletionMessage(
    #     content="""lambda row: row['gt'] == 'shield' """,
    #     stop_reason=StopReason.end_of_turn,
    # ), # todo fill this in with realistic context
]

code_gen_system_prompt = """
Generate a Python lambda function based on the provided context. The lambda function should represent a specific slice condition derived from the context. It must operate on a data row and only the `gt` (ground truth class) column, where the value represents object classes. Select only one ground truth class. Output only the lambda function in the format:

lambda row: row['gt'] == 'INSERT OBJECT CLASS NAME'
"""


# # Classes

# In[152]:


# load err analysis data

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import json


class ErrorAnalysisDataset(Dataset):
    def __init__(self, dataset_root, pred_split, img_root_dir=None, transform=None):

        self.data = pd.read_csv(os.path.join(dataset_root, "dataset.csv"))
        self.img_root_dir = img_root_dir
        self.transform = transform
        
        with open(os.path.join(dataset_root, 'mapping.json'), 'r') as f:
            self.mapping = json.load(f)
            idx_to_mapping = list(self.mapping)
            
        predictions = pd.read_csv(os.path.join(dataset_root, f"{pred_split}"), header=None, names=['pred'])
        self.data['pred']=predictions['pred'].values
        self.data['pred']=self.data['pred'].apply(lambda pred: self.mapping[idx_to_mapping[pred]])
        self.data['gt']=self.data['gt'].apply(lambda pred: self.mapping[idx_to_mapping[pred]])

        
    def __len__(self):
        return len(self.data)
        
    def load_image(self, img_path):
        img_path = os.path.join(self.img_root_dir, img_path)
        image = Image.open(img_path).convert('RGB')  # Ensure RGB
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load image
        img_path = os.path.join(self.img_root_dir, self.data.iloc[idx]['img_id'])
        image = Image.open(img_path).convert('RGB')  # Ensure RGB
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Extract metadata
        attribute = self.data.iloc[idx]['attribute']
        gt_code = self.data.iloc[idx]['gt_code']
        gt = self.data.iloc[idx]['gt']
        pred = self.data.iloc[idx]['pred']

        # Return a dictionary with the image and metadata
        sample = {
            'image': image,
            # 'attribute': attribute,
            # 'gt_code': gt_code,
            # 'gt': gt,
            # 'pred': pred
        }

        return sample


# In[153]:


# create a Context class that you can add to and read from, should be LLM ready
class Context:
    def __init__(self):
        self.context_strings = []
        self.actions = []
        self.action_assoc_with_context = []
        
    def add(self, string):
        if isinstance(string, list):
            string = str(string)
        self.context_strings.append(string)
        if len(self.actions) > 0:
            text = f'Step {len(self.actions)} [{self.actions[-1]}]: '
        else:
            text = f'Step {len(self.actions)} [Initial]: '


        self.action_assoc_with_context.append(text)
        
    def step(self, action):
        self.actions.append(action)
        
    def get_num_steps(self):
        return len(self.actions)
        
    def get_conclusion(self):
        if "conclusion" in self.actions[-1]:
            return self.context_strings[-1]
        else:
            return None
    def read(self):
        return '\n'.join(self.context_strings)
    def read_presentation(self):
        context_string_with_prefix = []
        for prefix, context in zip(self.action_assoc_with_context, self.context_strings):
            context_string_with_prefix.append(prefix+context)
        return '\n'.join(context_string_with_prefix)
        
    def read_exp(self):
        # read last three items
        return '\n'.join(self.context_strings[-4:])


# In[154]:


# confusion matrix analysis module
import numpy as np
class InitialAnalysis:
    def __init__(self, df, prediction_col, ground_truth_col, k=5):
        # Get unique classes
        classes = sorted(df[ground_truth_col].unique())
        
        # Initialize confusion matrix
        confusion_matrix = pd.DataFrame(
            np.zeros((len(classes), len(classes)), dtype=int),
            index=classes,
            columns=classes
        )
        
        # Populate the confusion matrix
        for _, row in df.iterrows():
            actual = row[ground_truth_col]
            predicted = row[prediction_col]
            confusion_matrix.loc[actual, predicted] += 1
        
        # Extract non-diagonal elements
        errors = []
        for actual in classes:
            for predicted in classes:
                if actual != predicted and confusion_matrix.loc[actual, predicted] > 0:
                    errors.append(((actual, predicted), confusion_matrix.loc[actual, predicted]))

        # Sort errors by count and take top k
        top_k_errors = sorted(errors, key=lambda x: x[1], reverse=True)[:k]

        self.confusion_matrix = confusion_matrix
        self.top_k_errors_pred_conditional = top_k_errors
        self.k = k
        self.classes = classes
        
    def human_readable_topk_pred_conditional_errors(self):
        errors_nl = []
        for (actual, predicted), err_count in self.top_k_errors_pred_conditional:
            errors_nl.append(f"The actual class is '{actual}', however model incorrectly predicts '{predicted}' {err_count} times")
            
        return '\n '.join(errors_nl)
    
    def human_readable_topk_errors_gt(self):
        marginal_errs = []
        for gt in self.classes:
            total = self.confusion_matrix.loc[gt].sum()
            marginal_errs.append((gt, (self.confusion_matrix.loc[gt].sum()-self.confusion_matrix.loc[gt, gt]).item()/total))

        # select top k
        marginal_errs = sorted(marginal_errs, key=lambda x: x[1], reverse=True)[:self.k]
        
        return f"The top five classes with the highest errors are: {marginal_errs}"


# In[155]:


def sample(data, slice_condition, prediction_col='pred', ground_truth_col='gt', n=10):
    # Filter data based on slice condition
    filtered_data = data[data.apply(slice_condition, axis=1)]
    
    # Divide into error set and non-error set
    error_set = filtered_data[filtered_data[prediction_col] != filtered_data[ground_truth_col]]
    non_error_set = filtered_data[filtered_data[prediction_col] == filtered_data[ground_truth_col]]

    # Sample n from both sets
    sampled_error = error_set.sample(n=min(len(error_set), n), random_state=42)  # Use random_state for reproducibility
    sampled_non_error = non_error_set.sample(n=min(len(non_error_set), n), random_state=42)

    # Return the sampled data with img_id, pred, and gt
    return sampled_error['img_id'].values, sampled_non_error['img_id'].values, sampled_error[[prediction_col, ground_truth_col]].values, sampled_non_error[[prediction_col, ground_truth_col]].values


# In[156]:


# captioning, prompt for caption Caption the image with the most salient details

def caption_sets(model, processor, dataset, err_list, non_error_list, err_pred_gt_list, non_error_pred_gt_list, system_prompt=global_caption_prompt, batch_size=1, device= 1 if MULTI_GPU_MODE else 0):
    images = []
    texts=[]
    images.extend([dataset.load_image(img_path) for img_path in err_list])
    images.extend([dataset.load_image(img_path) for img_path in non_error_list])

    for img in images:
        prompt = f"USER: <image>\n{system_prompt}\nASSISTANT:"
        texts.append(prompt)
    predictions = []

    if batch_size ==1:
        for img, txt in zip(images, texts):
            batch = processor(text=[txt], images=[img], return_tensors="pt", padding=True)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                   pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
            
            predictions.append(processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)[0])
    
    else:
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                               pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
        
        predictions = processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
    
    # err_list_prompt = [f'A \'{err_pred_gt_list[i][1]}\' sample with a incorrect prediction, \'{err_pred_gt_list[i][0]}\' has description: ' for i, _ in enumerate(err_list)]
    # non_err_list_prompt = [f'A \'{non_error_pred_gt_list[i][1]}\' sample with a correct prediction, \'{non_error_pred_gt_list[i][0]}\', has description: ' for i, _ in enumerate(non_error_list)]

    # removed the the incorrect sample name
    err_list_prompt = [f'A \'{err_pred_gt_list[i][1]}\' sample with a incorrect prediction has description: ' for i, _ in enumerate(err_list)]
    non_err_list_prompt = [f'A \'{non_error_pred_gt_list[i][1]}\' sample with a correct prediction has description: ' for i, _ in enumerate(non_error_list)]
    
    return [f'{text}{predictions[i]}' for i, text in enumerate([*err_list_prompt, *non_err_list_prompt])]


# In[157]:


# hypothesis + validator class
from fuzzywuzzy import fuzz

"""
Hypothesis Formulation:
Uses a LLM to formulate (1) a hypothesis on what might be a possible error-prone 
attribute given the context and (2) a labeling prompt given the generated hypothesis 
(e.g. given hypothesis "The model is likely misclassifying images in poorly lit 
conditions because details of the object are obscured.", the labeling prompt would be 
"For each image, label whether the lighting condition is: [1] 'Bright', [2] 'Dim', 
or [3] 'Dark'.")

Validation:
Using an M-LLM, the images in each set are labeled using the labeling prompt from the 
hypothesis formulation step (constrains the labeling space), and then error-rates are 
re-computed for each attribute in the labeled subset (e.g. given the hypothesis/prompt 
example above, the results could be Bright: 0.75 ER, Dim: 0.2 ER, Dark: 0.25 ER; note 
this doesn’t need to sum to 1) and this result is passed to the context.
"""
import re

class ExperimentFlow:
    def __init__(self, llm, mllm=None, verbose=False, code_gen=None):
        self.llm = llm
        self.mllm = mllm
        self.verbose = verbose
        self.code_gen = code_gen

    def form_hypothesis(self, context, 
                        prompt=global_hypothesis_system_prompt):
        
        dialog = [
                SystemMessage(content=prompt),
                *global_hypothesis_few_shot_example,
                UserMessage(content=f'\nContext: {context}\nHypothesis: '),
            ]
        result = self.llm.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        if self.verbose:
            for msg in dialog:
                print(f"{msg.role.capitalize()}: {msg.content}\n")
    
        return result.generation.content

    def get_labeling_prompt(self, hypothesis, prompt=global_labeling_system_prompt):
        dialog = [
                SystemMessage(content=prompt),
                *global_labeling_few_shot,
                UserMessage(content=f'Hypothesis: {hypothesis}\nLabeling Prompt: ')
        ]

        result = self.llm.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        if self.verbose:
            for msg in dialog:
                print(f"{msg.role.capitalize()}: {msg.content}\n")
    
        return result.generation.content
    def label_images(self, image_list, labeling_prompt, device = 1 if MULTI_GPU_MODE else 0):
        images = [self.code_gen.dataset.load_image(img_path) for img_path in image_list]
        # texts=[]
        predictions = []
        # self.mllm['model'].eval()

        # batch size 1 due to GPU mem limit with LLM
        for img in images:
            prompt = f"USER: <image>\n{global_label_images_system_prompt} {labeling_prompt}\nASSISTANT:"
            # texts.append(prompt)
    
            batch = self.mllm['processor'](text=[prompt], images=[img], return_tensors="pt", padding=True)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
        
            generated_ids = self.mllm['model'].generate(input_ids=input_ids, attention_mask=attention_mask,
                                                   pixel_values=pixel_values, max_new_tokens=100)
            
            pred = self.mllm['processor'].batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
            predictions.append(pred[0])
        return predictions
    
    def validate(self, labeling_prompt, error_img_list, correct_img_list):
        def parse_categories(labeling_prompt):
            # Extract categories from the prompt
            categories = []
            lines = labeling_prompt.split("\n")
            for line in lines:
                if "[" in line and "]" in line:
                    # Extract text between [number] and the label
                    category = line.split("]")[1].strip(" '")
                    if ":" in line:
                        category=category.split(":")[0]
                    categories.append(category)
            return categories
            
        def normalize(text):
            return re.sub(r'\W+', ' ', text.lower().strip())

        def iou(prediction, ground_truth):
            tokenized_prediction = set(normalize(prediction).split())
            tokenized_ground_truth = set(normalize(ground_truth).split())
            return len(tokenized_prediction.intersection(tokenized_ground_truth)) / len(tokenized_ground_truth)
        def simplify_cat(cat):
            return re.sub(r'\s*\(.*?\)', '', cat) 
        # Transformed categories
        # categories = simplify_categories(categories)
        def get_match(prediction, categories):
            prediction = simplify_cat(prediction)
            best_match = max(categories, key=lambda cat: fuzz.partial_ratio(prediction, cat))
            if fuzz.partial_ratio(prediction, best_match) > 70:  # Example threshold
                return best_match
            return "Unknown"
            
        def calc_err_rate(attribute_classifications, err_list, categories):
            # 1 represents error in err_list
            category_errors = {category: 0 for category in categories}
            total_count = {category: 0 for category in categories}
        
            # Calculate errors and counts for each category
            for i, classification in enumerate(attribute_classifications):
                if classification in categories:
                    total_count[classification] += 1
                    if i < len(err_list)//2:  # Error list indices
                        category_errors[classification] += 1
            
            # Calculate error rate per category
            error_rates = {
                category: (category_errors[category] / total_count[category]) if total_count[category] > 0 else 0
                for category in categories
            }
            return error_rates
            
        categories = parse_categories(labeling_prompt)
        categories.append("Unknown")
        categories=[simplify_cat(cat) for cat in categories]
        print(categories)
        predictions = self.label_images([*error_img_list, *correct_img_list], labeling_prompt)
        print("predictions")
        print(predictions)
        attribute_classifications=[]
        for i, _ in enumerate(error_img_list):
            attribute_classifications.append(get_match(predictions[i], categories))
        for i, _ in enumerate(correct_img_list):
            attribute_classifications.append(get_match(predictions[i+len(error_img_list)], categories))

        print("attribute classifications")
        print(attribute_classifications)
        
        error_rates = calc_err_rate(attribute_classifications, [1 if i < len(error_img_list) else 0 for i in range(len(error_img_list)+len(correct_img_list))], categories)

        # todo: validate hypothesis based on err_rates calculated, maybe leave for later?
        return error_rates

    def run(self, context):
        hypothesis = self.form_hypothesis(context.read())
        print("Hypothesis for experiment: {}".format(hypothesis))
        context.add(hypothesis)
        labeling_prompt = self.get_labeling_prompt(hypothesis)
        print("Labeling prompt for experiment: {}".format(labeling_prompt))
        context.add(labeling_prompt)
        slice_condition = self.code_gen.get_slice_condition(context.read())
        context.add("Slice Condition for experiment: {}".format(slice_condition))
        print("Slice Condition for experiment:", slice_condition)
        try:
            err_list, non_error_list, _, _ = sample(self.code_gen.dataset.data, eval(slice_condition))
        except:
            raise Exception("Invalid Slice Condition")

        result = self.validate(labeling_prompt, err_list, non_error_list)
        context.add("Error Rates for experiment: {}".format(result))
        return context


# In[158]:


# llm brain class

class LLMBrain:
    def __init__(self, llm, verbose=False):
        self.llm = llm
        self.verbose = verbose
    def conclusion(self, context, prompt="Conclude the error slice based on the context"):
        pass

    def prune(self, reasoning):
        dialog = [*global_prune_labeling_output_prompt,
                    UserMessage(content=f"Reasoning: {reasoning}\nSelected Action:"),
                 ]


        result = self.llm.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        if self.verbose:
            for msg in dialog:
                print(f"{msg.role.capitalize()}: {msg.content}\n")
        action = result.generation.content
        print("In prune", action)
        if "gather_observations" in action:
            return "gather_observations"
        elif "conclusion" in action:
            return "conclusion"
        elif "experiment" in action:
            return "experiment"
        return action # should be an issue
                
    def agent(self, context, prompt = global_agent_prompt):
        dialog = [
                SystemMessage(content=prompt),
                *global_agent_few_shot_prompt,
                # TODO: Add examples
                # UserMessage(content=f' '),
                # CompletionMessage(
                #     content=""" """,
                #     stop_reason=StopReason.end_of_turn,
                # ),
                UserMessage(content=f'Context: {context}\nSelected Action: ')
        ]

        result = self.llm.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        if self.verbose:
            for msg in dialog:
                print(f"{msg.role.capitalize()}: {msg.content}\n")

        reasoning = result.generation.content
        action = self.prune(reasoning)

        return reasoning, action
        


# # Load Models

# In[13]:


# load mllm model
MAX_LENGTH = 384
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right
device = 1 if MULTI_GPU_MODE else 0
model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        # _attn_implementation="flash_attention_2",
).to(device)


# In[14]:


# load LLM
import os

# Set the master address and port
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
from llama_models.llama3.reference_impl.generation import Llama
from llama_models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)

temperature = 0.6
top_p = 0.9
max_seq_len = 10000
max_batch_size = 4
max_gen_len = None
model_parallel_size = None
tokenizer_path = str("/ix/cs3550_2024f/ErrAttr-ID/tokenizer.model")
generator = Llama.build(
    ckpt_dir="/ix/cs3550_2024f/ErrAttr-ID/Llama3.1-8B-Instruct",
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    model_parallel_size=model_parallel_size,
)


# In[15]:


model.eval()
generator.model.eval()


class CodeGen:
    def __init__(self, llm, dataset, mllm, system_prompt=code_gen_system_prompt, verbose=False):
        self.llm = llm
        self.system_prompt = system_prompt
        self.verbose=False
        self.dataset=dataset
        self.mllm = mllm
    def take_action(self, context, action):
        if "gather_observations" in action:
            slice_condition = self.get_slice_condition(context.read())
            print(slice_condition)
            context.add(slice_condition)
            err_list, non_error_list, err_pred_gt_list, non_error_pred_gt_list = sample(self.dataset.data, eval(slice_condition))
            result = caption_sets(self.mllm['model'], self.mllm['processor'], self.dataset, err_list, non_error_list, err_pred_gt_list, non_error_pred_gt_list)
        elif "conclusion" in action:
            return "DONE"
        elif "experiment" in action:
            experiment = ExperimentFlow(generator, mllm=self.mllm, code_gen=self)
            result = experiment.run(context)
        else:
            raise Exception(f"Action '{action}' is not in set of okay actions=['gather_observations', 'conclusion', 'experiment']")
        return result
    def get_slice_condition(self,context, prompt=slice_condition_prompt):
        dialog = [
                SystemMessage(content=self.system_prompt),
                *prompt,
                UserMessage(content=f'Context: {context}\n Slice Condition Code: ')
        ]

        result = self.llm.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        if self.verbose:
            for msg in dialog:
                print(f"{msg.role.capitalize()}: {msg.content}\n")
    
        return result.generation.content

def main(context, split_file_id='pred_splits/split_1.txt', verbose=False, max_steps=10, retry=False):
    step=0
    step_str='Step {}: {}' #step_str.format(step)

    def log(output):
        print(step_str.format(step, output))
        
    # context=Context()
    dataset=ErrorAnalysisDataset(dataset_root='../mock_data_creation/mock_data', pred_split=split_file_id, img_root_dir='/ix/akovashka/arr159/imagenet-r')
    analysis = InitialAnalysis(dataset.data, prediction_col='pred', ground_truth_col='gt')
    context.add(analysis.human_readable_topk_errors_gt())
    # context.add(analysis.human_readable_topk_pred_conditional_errors())
    if verbose:
        log(context.read())
    brain = LLMBrain(generator)
    step+=1
    with torch.no_grad():

        first_action_reasoning, action = brain.agent(context.read())
    if verbose:
        log(first_action_reasoning)
        log(f"Selected action: {action}")
        
    context.add(first_action_reasoning)
    context.step(action)
    code_gen = CodeGen(generator, dataset, mllm={'processor': processor, 'model':model})
    output = first_action_reasoning
    previous_outputs = Context()
    while step < max_steps:
        with torch.no_grad():

            output = code_gen.take_action(context, action)
        step+=1
        if output=="DONE":
            log(output)
            # context.add(output)
            break
        elif isinstance(output, Context):
            log(output.read_exp()) # it was an exp
        else:
            log(output)
            context.add(output)
            
        with torch.no_grad():

            output, action = brain.agent(context.read())
            context.step(action)

        step+=1

        if verbose:
            log(output)
            
        context.add(output)


import pandas as pd
import gc
import torch
def evaluate(path_to_eval, path_to_splits='demo_pred_splits', n = 10, skip=2):
    # load df
    eval_df = pd.read_csv(path_to_eval)

    if n != len(eval_df):
        eval_df = eval_df.sample(n=n, random_state=42).iloc[skip:]
        
    errors=[]
    steps = []
    action_list = []
    conclusions = []
    contexts=[]
    gt_successful=[]
    gt_all=[]
    for _, row in eval_df.iterrows():
        _, attribute, object_name, file_name = row
        print(attribute, object_name)
        gt_all.append((attribute, object_name))
        context=Context()
        try:
            main(context, split_file_id=f'{path_to_splits}/{file_name}', max_steps=6, verbose=True)
            # print("Num Steps:", context.get_num_steps())
            # print("Actions:", context.actions)
            # print("Conclusion, if any:", context.get_conclusion())
            steps.append(context.get_num_steps())
            action_list.append(context.actions)
            conclusions.append(context.get_conclusion())
            gt_successful.append((attribute, object_name))
        except Exception as e:
            errors.append(e)
        gc.collect()
        torch.cuda.empty_cache()
        contexts.append(context)
    print("\n--- Stats ---")
    print(f"Number of errors: {len(errors)}")
    print(f"Total number of steps: {sum(steps)}")
    print(f"Average number of steps per successful run: {sum(steps) / len(steps) if steps else 0}")
    
    from collections import Counter
    flat_action_list = [action for sublist in action_list for action in sublist]

    # Use Counter on the flattened list
    action_distribution = Counter(flat_action_list)
    
    # Print action distribution
    print(f"Distribution of actions:")
    for action, count in action_distribution.items():
        print(f"  {action}: {count}")
    
    non_none_conclusions = sum(1 for conclusion in conclusions if conclusion is not None)
    print(f"Number of conclusions that are not None: {non_none_conclusions}")
    
    return contexts, errors, steps, action_list, conclusions, gt_all, gt_successful


# In[165]:


import time

# Start the timer
start_time = time.time()
contexts, errors, steps, action_list, conclusions, gt_all, gt_successful = evaluate('../mock_data_creation/mock_data/eval_df.csv', path_to_splits='demo_pred_splits', n = 202)


# In[166]:


# End the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")


# In[167]:


print("\n--- Stats ---")
print(f"Number of errors: {len(errors)}")
print(f"Total number of steps: {sum(steps)}")
print(f"Average number of steps per successful run: {sum(steps) / len(steps) if steps else 0}")

from collections import Counter
flat_action_list = [action for sublist in action_list for action in sublist]

# Use Counter on the flattened list
action_distribution = Counter(flat_action_list)

# Print action distribution
print(f"Distribution of actions:")
for action, count in action_distribution.items():
    print(f"  {action}: {count}")

non_none_conclusions = sum(1 for conclusion in conclusions if conclusion is not None)
print(f"Number of conclusions that are not None: {non_none_conclusions}")


import pickle

# Data to save
data = {
    "contexts": contexts,
    "errors": errors,
    "steps": steps,
    "action_list": action_list,
    "conclusions": conclusions,
    "gt_all": gt_all,
    "gt_successful": gt_successful
}

# Save to a pickle file
with open("eval_outputs/saved_outputn=202.pkl", "wb") as file:
    pickle.dump(data, file)

print("Data saved to eval_outputs/saved_outputn=202.pkl")




