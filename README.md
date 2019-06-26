# QBASR
This is a paper that investigates question answering under conditions of noisy input.  
1. We experiment with generating a large synthetic corpus and compare this with accuracy on human recorded data.
2. We introduce a confidence model that directly incorporates the confidence from an automatic-speech-recognition (ASR) system into a question answering neural network.
3. We compare model accuracy on data with "unknowns" decoded by the ASR system and data where a known vocabulary prediction is always forced. 

## Paper

The paper can be found at:
LINK_TO_BE_UPDATED

## Code
Code for the Deep Averaging Network and the Infromation Retrieval system is provided here.  Additional data generation code can be found at 
https://github.com/DenisPeskov/QBASR_GenerateData

## Data

Data is stored at http://qbasr.umiacs.io/
There are three folders:
1. Human Original Data
2. QANTA
3. SearchQA

Human original data contains the .mp3s for (and .wav, .lat, and .sau) files needed for generating the final text output.  The processed text files from this data are stored in the respective place below.  

QANTA contains the processed text data for Quizbowl.

SearchQA contains the processed text data for Jeopardy.

Auto generated data is not stored in the interest of space (~300 GB). A seperate repository containing the code needed to generate audio data with Google Text to Speech and decode the data with Kaldi is provided at:
https://github.com/DenisPeskov/QBASR_GenerateData

###  Example of Quizbowl data (the same sentence across different speakers):
1. Speaker 1: http://qbasr.umiacs.io/HumanOriginalData/HumanData_MultiSpeaker/0/2_0.mp3
2. Speaker 2: http://qbasr.umiacs.io/HumanOriginalData/HumanData_MultiSpeaker/1/2_0.mp3
3. Auto Generated: http://qbasr.umiacs.io/HumanOriginalData/HumanData_MultiSpeaker/999/-997_0.mp3

### Example of Jeopardy data:

Alex Trebek's voice http://qbasr.umiacs.io/HumanOriginalData/HumanData_9.26.18_FullJeopardyGame/2_1.mp3

Jeopardy Show #7828 (http://www.j-archive.com/showgame.php?game_id=6112) is hand-parsed.  

Data is stored as: columnNumber_QuestionNumber  (with 1 not 0 index).  So 2_1 corresponds to the first question second column (Name the Novel): "If the picture was to alter, it was to alter. That was all...not one blossom of his loveliness would ever fade".
