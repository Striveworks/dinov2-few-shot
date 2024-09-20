import gradio as gr

from image_classifier.svm import train_svm, predict_image, save_svm

# Initialize global SVM variable
svm_classifier = None

if __name__ == "__main__":
    # Gradio interface
    with gr.Blocks() as demo:
        # Image folder input for training
        folder_input = gr.File(
            label="Upload image folder (folder/class_name)",
            file_count="directory",
            height=512,
        )
        val_split_input = gr.Slider(
            0.1, 0.5, step=0.05, value=0.2, label="Validation Split Size"
        )
        train_button = gr.Button("Train SVM")
        train_output = gr.Textbox(label="Training output")

        # Image input for prediction
        image_input = gr.Image(label="Upload an image for prediction")
        predict_button = gr.Button("Predict")
        predict_output = gr.Textbox(label="Prediction")

        # Download button for trained SVM
        download_button = gr.Button("Download Trained SVM")
        download_output = gr.File(label="Download SVM")

        # Train SVM
        train_button.click(
            train_svm, inputs=[folder_input, val_split_input], outputs=train_output
        )

        # Predict image
        predict_button.click(predict_image, inputs=image_input, outputs=predict_output)

        # Download SVM
        download_button.click(save_svm, outputs=download_output)

    # Launch the Gradio app
    demo.launch()
