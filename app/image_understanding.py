import ollama


# ============================================================
# VISION MODEL
# ============================================================

# Use LLaVA because llama3.2-vision is currently failing
# with the mllama architecture error in your Ollama setup.
VISION_MODEL = "llava"


def describe_image(image_path, ocr_text=""):
    """
    Analyze an image using the Ollama LLaVA vision model.

    The actual image is provided to the model along with OCR
    text extracted from the same image.

    The model is instructed to describe only information
    supported by the image and OCR.
    """

    prompt = """
You are the image-understanding component of NoteMind.

Analyze the provided image carefully.

Your task is to explain what is VISUALLY present in the image.

Use the OCR text only as additional context.
Do not blindly trust OCR if it is clearly incorrect.

Pay particular attention to:

- diagrams
- figures
- process flow
- arrows
- labels
- relationships between objects
- tables
- charts
- graphs
- code shown in the image
- sequence or timing
- communication between components
- inputs and outputs
- important visual structure

If the image is a diagram or figure:

1. Identify what the figure represents.
2. Identify the main components.
3. Explain the relationships between the components.
4. Explain arrows, directions, or communication paths.
5. Explain any sequence or timing shown.
6. Explain the purpose of the figure if it can be determined
   from the image and OCR/context.

Do not invent information that cannot be determined.

Keep the explanation concise but useful for a question-answering
system such as NoteMind.

OCR extracted from the image:

""" + (ocr_text or "No OCR text available.")

    try:

        print(
            f"  Using vision model: {VISION_MODEL}",
            flush=True
        )

        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ]
        )

        answer = response["message"]["content"].strip()

        if answer:
            print(
                "  Vision analysis completed.",
                flush=True
            )

        return answer

    except Exception as error:

        print(
            f"  Warning: Image understanding failed: {error}",
            flush=True
        )

        return ""
