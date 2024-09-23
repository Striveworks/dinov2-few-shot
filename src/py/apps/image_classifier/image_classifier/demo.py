import gradio as gr

from image_classifier.svm import train_svm, predict_image, save_svm

# Initialize global SVM variable
svm_classifier = None

def upload_and_preserve_structure(files):
    """
    Reconstructs the folder structure of the uploaded files.

    Parameters
    ----------
    files : list
        List of uploaded file objects.

    Returns
    -------
    dict
        Dictionary mapping the folder names to the files in that folder.
    """
    folder_structure = {}
    
    for file_obj in files:
        # Get the original file path
        original_path = Path(file_obj.orig_name)
        
        # Get the relative path (simulate preserving the folder structure)
        relative_folder = str(original_path.parent)
        if relative_folder not in folder_structure:
            folder_structure[relative_folder] = []
        
        # Save the file in the simulated folder structure
        folder_structure[relative_folder].append(file_obj.name)
    
    return folder_structure

def display_structure(files):
    """
    Displays the folder structure of uploaded files.

    Parameters
    ----------
    files : list
        List of uploaded file objects.

    Returns
    -------
    str
        Formatted folder structure.
    """
    structure = upload_and_preserve_structure(files)
    output = "Uploaded Folder Structure:\n"
    for folder, file_list in structure.items():
        output += f"Folder: {folder}\n"
        for file in file_list:
            output += f"  - {file}\n"
    return output

if __name__ == "__main__":
    # Gradio interface
    with gr.Blocks() as demo:
        # Image folder input for training
        # folder_input = gr.File(
        #     label="Upload image folder (folder/class_name)",
        #     file_count="directory",
        #     height=512,
        # )
        folder_input = gr.Files(label="Upload Files with Folder Structure", file_count="multiple", height=512,)
        upload_button = gr.Button("Upload")
        structure_output = gr.Textbox(label="Folder Structure", interactive=False)
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
    demo.launch(server_name="0.0.0.0", server_port=8000)
