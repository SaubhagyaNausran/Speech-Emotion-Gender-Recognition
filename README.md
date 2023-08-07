# Speech-Emotion-Gender-Recognition

## Overview
This project aims to recognize the emotion and gender of a speaker from their speech. It uses a Conv1D model for the recognition process, and the trained model achieves an accuracy of 93%.

## Key Files and Directories
- `Emotion_Model_conv1d_gender_93.h5`: Contains the trained model weights.
- `Emotion_Model_conv1d_gender_93.json`: Contains the model architecture in JSON format.
- `New_Features.csv`: A CSV file containing features extracted from the audio data.
- `app.py`: The main application script that runs the web interface for the project.
- `audio.wav`: Sample audio file.
- `requirements.txt`: Lists all the dependencies required to run the project.
- `templates/`: Contains the HTML templates used in the web interface.

## How to Run
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application using `python app.py`.
4. Access the web interface through your browser and upload an audio file to get the emotion and gender prediction.

## Future Enhancements
- Improve the model accuracy by using a larger dataset.
- Integrate real-time audio recording and prediction in the web interface.

    second = data.get('second', 0)
    result = first - second
    return jsonify({"result": result})

if _name_ == '_main_':
    app.run(port=8080, host='0.0.0.0')
