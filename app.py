import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "saved_roberta_student_distilled"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict(text):
    lines = text.strip().split("\n")
    inputs = tokenizer(lines, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probs, dim=1).tolist()
        confidences = probs.max(dim=1).values.tolist()

    line_results = [
        f"[{'AI' if p == 1 else 'Human'} - {round(c, 3)}] {line}"
        for line, p, c in zip(lines, predictions, confidences)
    ]
    overall_label = "AI" if predictions.count(1) > predictions.count(0) else "Human"
    return f"Overall Prediction: {overall_label}\n\nLine-by-line:\n" + "\n".join(line_results)

demo = gr.Interface(fn=predict, inputs="textbox", outputs="textbox", title="AI vs Human Text Detector")
demo.launch()
