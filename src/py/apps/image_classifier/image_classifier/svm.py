from image_classifier.embedding import get_embeddings_and_labels

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from PIL import Image


def train_svm(image_folder: str, val_split_size: float = 0.1, kernel: str = "rbf"):
    """
    Trains an SVM classifier on images in the given folder using DINOv2 embeddings and splits data for validation.

    Parameters
    ----------
    image_folder : str
        The path to the folder containing subfolders with class names.
    val_split_size : float
        The proportion of the data to include in the validation split.
    kernel : str
        The kernel for the SVM classifier. Defaults to radial basis functions.
        See sklearn docs for valid kernel choices.

    Returns
    -------
    str
        A message indicating the result of the training process and validation accuracy.
    """
    global svm_classifier

    embeddings, labels, label_idxs, image_paths = get_embeddings_and_labels(
        image_folder
    )

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=val_split_size, stratify=labels
    )

    # Train SVM
    svm_classifier = svm.SVC(probability=True, kernel=kernel)
    svm_classifier.fit(X_train, y_train)

    # Validation accuracy
    y_pred = svm_classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return f"SVM training completed! Validation accuracy: {accuracy * 100:.2f}%"


def predict_image(image: Image):
    """
    Predicts the class of an input image using the trained SVM classifier.

    Parameters
    ----------
    image : PIL.Image
        The image to predict.

    Returns
    -------
    str
        The predicted class name or an error message if the SVM is not trained.
    """
    global svm_classifier

    if svm_classifier is None:
        return "SVM not trained yet!"

    embedding = extract_embedding(image)
    prediction = svm_classifier.predict(embedding)[0]
    return prediction


def save_svm():
    """
    Saves the trained SVM classifier to a file.

    Returns
    -------
    str or None
        The filename of the saved SVM model, or None if the model is not trained.
    """
    global svm_classifier
    if svm_classifier is not None:
        with open("trained_svm.pkl", "wb") as f:
            pickle.dump(svm_classifier, f)
        return "trained_svm.pkl"
    return None
