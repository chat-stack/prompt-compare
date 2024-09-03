import openai
import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Constants for default values
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 150
DEFAULT_TOP_P = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
NUM_PAIRS = 2  # This means creating two side-by-side input/output boxes

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def generate_responses(prompts, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    results = []
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            result = response['choices'][0]['message']['content']
        except Exception as e:
            result = f"An error occurred: {str(e)}"
        results.append(result)
    return results

def gradio_interface(*args):
    prompts = args[:NUM_PAIRS]
    model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty = args[NUM_PAIRS:]
    
    if not any(prompts):
        return ["Please enter at least one prompt."] * NUM_PAIRS
    
    return generate_responses(prompts, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty)

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## OpenAI Chat Bulk Evaluator")
    
    model = gr.Textbox(label="Model", value=DEFAULT_MODEL, placeholder="e.g., gpt-4o-mini, gpt-4o")
    temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=DEFAULT_TEMPERATURE, step=0.1)
    max_tokens = gr.Slider(label="Max Tokens", minimum=1, maximum=4096, value=DEFAULT_MAX_TOKENS, step=1)
    top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=DEFAULT_TOP_P, step=0.1)
    frequency_penalty = gr.Slider(label="Frequency Penalty", minimum=-2.0, maximum=2.0, value=DEFAULT_FREQUENCY_PENALTY, step=0.1)
    presence_penalty = gr.Slider(label="Presence Penalty", minimum=-2.0, maximum=2.0, value=DEFAULT_PRESENCE_PENALTY, step=0.1)

    input_boxes = []
    output_boxes = []
    
    # Row for prompt input boxes
    with gr.Row():
        for i in range(NUM_PAIRS):
            input_box = gr.Textbox(label=f"Prompt {i+1}", lines=5, placeholder=f"Enter prompt {i+1}")
            input_boxes.append(input_box)
    
    # Row for response output boxes
    with gr.Row():
        for i in range(NUM_PAIRS):
            output_box = gr.Textbox(label=f"Response {i+1}", lines=5)
            output_boxes.append(output_box)

    submit_button = gr.Button("Generate Responses")

    submit_button.click(
        fn=gradio_interface, 
        inputs=[*input_boxes, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty], 
        outputs=output_boxes
    )

# Launch the Gradio interface
demo.launch()