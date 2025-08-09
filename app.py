import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_path = "saved_roberta_student_distilled"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prediction function
def predict(text):
    if not text.strip():
        return "❌ Please enter some text.", None

    lines = text.strip().split("\n")
    inputs = tokenizer(lines, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probs, dim=1).tolist()
        confidences = probs.max(dim=1).values.tolist()

    # Overall prediction
    overall_label = "AI" if predictions.count(1) > predictions.count(0) else "Human"

    # Make a nicer HTML output
    results_html = f"<h2>Overall Prediction: <span style='color:{'red' if overall_label=='AI' else 'green'}'>{overall_label}</span></h2>"
    results_html += "<br><h3>Line-by-line Analysis:</h3>"

    for line, pred, conf in zip(lines, predictions, confidences):
        label = "AI" if pred == 1 else "Human"
        color = "red" if label == "AI" else "green"
        results_html += f"<p><b style='color:{color}'>{label}</b> ({conf:.2%})<br>{line}</p>"
        # Optional confidence bar
        results_html += f"<div style='background:#ddd;width:100%;height:8px;border-radius:4px;'>"
        results_html += f"<div style='background:{color};width:{conf*100}%;height:8px;border-radius:4px;'></div></div><br>"

    return results_html, None

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🕵️ AI vs Human Text Detector")
    gr.Markdown("Paste any text below to check if it was written by AI or a human.")

    input_box = gr.Textbox(lines=8, placeholder="Paste your text here...")
    output_html = gr.HTML()
    run_btn = gr.Button("Analyze Text")

    run_btn.click(predict, inputs=input_box, outputs=output_html)

demo.launch()
