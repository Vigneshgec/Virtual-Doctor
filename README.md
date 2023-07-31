# Medical Diagnosis Web Application

This is a web application for medical diagnosis based on user-entered symptoms and text extracted from uploaded images. The application utilizes machine learning models and natural language processing techniques to provide diagnoses and treatment recommendations.

## Installation

To run the application locally, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/medical-diagnosis.git
   ```
   
2. Install the required dependencies. Make sure you have Python 3 and pip installed:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` file contains the necessary libraries and their versions for this application.
   
4. Download and install Tesseract OCR:
   - **Windows**: Download the installer from the Tesseract OCR website and follow the installation instructions.
   - **Linux**: Use your package manager to install Tesseract OCR. For example, on Ubuntu, you can run the following command:
     ```bash
     sudo apt-get install tesseract-ocr
       ```

5. Run the application:
   ```bash
   python app.py
   ```
   The application will be accessible at `http://localhost:5000` in your web browser.

## Usage
Open the web application in your web browser.
1. Upload an image containing relevant medical information. The application will extract text from the image using OCR.
2. Enter your symptoms in the provided input field. Separate multiple symptoms with commas.
3. Click the "Get Diagnosis" button to receive the diagnosis and treatment recommendations.
The application will display the diagnosis, recommendation, and a health score based on the severity of the predicted diagnosis. If the health score is below 20, a link to make an appointment will be provided.

## Dataset
The application uses a pre-existing dataset containing symptoms, diagnoses, treatment recommendations, and severity scores. The dataset is stored in a CSV file (`data1.csv`) and is used for training the machine learning model.

## Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgments
This application was developed using various open-source libraries and resources. Special thanks to the authors and contributors of the following:
- Flask
- NumPy
- Pandas
- NLTK
- Scikit-learn
- OpenCV
- Tesseract OCR
- Spacy

## Contact
If you have any questions or inquiries, please contact vigneshmodepalli@gmail.com.
