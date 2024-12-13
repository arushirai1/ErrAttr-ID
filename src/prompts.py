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


global_labeling_system_prompt ="Create a extremely concise labeling prompt given a hypothesis"
global_labeling_few_shot = [UserMessage(content=f'Hypothesis: The model is likely misclassifying images in poorly lit conditions because details of the object are obscured.\nLabeling Prompt: '),
                CompletionMessage(
                    content="""For each image, label from two options whether the lighting condition is: [1] 'Bright', [2] 'Dim/Dark'.""",
                    stop_reason=StopReason.end_of_turn,
                ),
        ]

global_label_images_system_prompt =""

global_prune_labeling_output_prompt = [
                SystemMessage(content="Given the reasoning, only generate the best match: gather_evidence, experiment, conclusion"),
                # TODO: Add examples
                UserMessage(content=f'Based on the context, the selected action is:\n\n(1) gather_evidence\n\nThe context provides information on classes with the highest errors but does not provide enough information to conclude the error slice (specific attributes of erronous images/objects) or form a hypothesis that requires experimentation. Therefore, the next step is to gather more evidence to analyze the error slice further.'),
                CompletionMessage(
                    content="""gather_evidence""",
                    stop_reason=StopReason.end_of_turn,
                ),
        ] # this is for prune() in LLM brain


global_agent_prompt = '''The goal of this system is to find specific error-slice attributes such as properties of an object or image. Given context, select ONE ACTION from the following actions: gather_evidence, experiment, or conclusion.
                                        (1) gather_evidence: given the context, gather more descriptions for a particular object class.
                                        (2) experiment: given some context and evidence, you need to form a hypothesis and evaluate to confirm. generally do experiment before conclusion
                                        (3) conclusion: if there is enough information in the context to conclude the error slice
                                        '''

# only need slice condition returned from context as code_gen

slice_condition_prompt = [
    UserMessage(
        content='''Context: "The top five marginal errors are for these classes: [('cannon', np.float64(0.3698630136986301)), ('shield', np.float64(0.36496350364963503)), ('lipstick', np.float64(0.34177215189873417)), ('gazelle', np.float64(0.3381294964028777)), ('baseball_player', np.float64(0.32941176470588235))]"
        'Based on the context, the selected action is:\n\n(1) gather_evidence\n\nThe context provides information about the top five marginal errors, descriptions of samples with incorrect predictions, and descriptions of samples with correct predictions. However, there is no clear indication of what specific error-slice attribute (property of an object or image) is being sought. Therefore, gathering evidence is the most appropriate action to take next, as it would involve examining the descriptions and images further to identify any patterns or characteristics that may be contributing to the errors.',
        Slice Condition Code: '''),
    CompletionMessage(
        content="""lambda row: row['gt'] == 'cannon' """,
        stop_reason=StopReason.end_of_turn,
    ),
    UserMessage(
        content='''Context: "The top five marginal errors are for these classes: [('cannon', np.float64(0.3698630136986301)), ('shield', np.float64(0.36496350364963503)), ('lipstick', np.float64(0.34177215189873417)), ('gazelle', np.float64(0.3381294964028777)), ('baseball_player', np.float64(0.32941176470588235))]"
        'Based on the context, the selected action is:\n\n(1) gather_evidence\n\nThe context provides information about the top five marginal errors, descriptions of samples with incorrect predictions, and descriptions of samples with correct predictions. However, there is no clear indication of what specific error-slice attribute (property of an object or image) is being sought. Therefore, gathering evidence is the most appropriate action to take next, as it would involve examining the descriptions and images further to identify any patterns or characteristics that may be contributing to the errors.',
        Previous slice_condition = lambda row: row['gt'] == 'cannon'
        "Results of investigating cannon experiment: {'**Cartoon/Abstract**': 0.5, '**Realistic/High-Res**': 0.5, '**Other**': 0, 'Unknown': 0.5}"
        There is not enough information to conclude, so we will gather evidence again except for a different class
        '''),
    CompletionMessage(
        content="""lambda row: row['gt'] == 'shield' """,
        stop_reason=StopReason.end_of_turn,
    ), # todo fill this in with realistic context
]

code_gen_system_prompt = """
Generate a Python lambda function based on the provided context. The lambda function should represent a specific slice condition derived from the context. It must operate on a data row containing only `pred` (predicted class) and `gt` (ground truth class) columns, where both values represent object classes. Focus on identifying conditions that match the context, such as specific classes or combinations of classes. Output only the lambda function in the format:

lambda row: <condition>
"""