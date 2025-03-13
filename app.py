import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import ollama
from pydantic import BaseModel


def handle_csv_upload(file):
    try:
        df = pd.read_csv(file)
        return df.head()
    except Exception as e:
        return f"Error: {str(e)}"


class QueryRequest(BaseModel):
    prompt: str

def ask_question(prompt, file):
    try:
        df = pd.read_csv(file)
        request = QueryRequest(prompt=prompt)

        
        model_name = "llama-2-3b-quantized"  

        
        try:
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": request.prompt}])
            return response['content']
        except Exception as model_error:
            return f"Model loading failed: {str(model_error)}"

    except Exception as e:
        return f"Error in processing: {str(e)}"

# Graph plotting
def plot_graph(file, x_col, y_col):
    try:
        df = pd.read_csv(file)
        plt.figure(figsize=(8,6))
        plt.plot(df[x_col], df[y_col], marker='o')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.grid(True)
        plt.tight_layout()
        return plt
    except Exception as e:
        return f"Error in plotting: {str(e)}"

# Build the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## CSV Question Answering and Visualization App")
    
    with gr.Tab("Upload and Preview"):
        file_input = gr.File(label="Upload CSV")
        preview_output = gr.Dataframe()
        file_input.change(fn=handle_csv_upload, inputs=file_input, outputs=preview_output)
    
    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(label="Enter your question")
        answer_output = gr.Textbox()
        question_input.submit(fn=ask_question, inputs=[question_input, file_input], outputs=answer_output)
    
    with gr.Tab("Plot Graphs"):
        x_col_input = gr.Dropdown(choices=[], label="X-axis column")
        y_col_input = gr.Dropdown(choices=[], label="Y-axis column")
        plot_output = gr.Plot()
        file_input.change(fn=lambda f: list(pd.read_csv(f).columns), inputs=file_input, outputs=[x_col_input, y_col_input])
        gr.Button("Generate Plot").click(fn=plot_graph, inputs=[file_input, x_col_input, y_col_input], outputs=plot_output)

app.launch(share=True)










