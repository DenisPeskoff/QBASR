# QBASR



## Data

Data is stored at http://qbasr.umiacs.io/
There are three folders:
### Human Original Data
### QANTA
### SearchQA

Human original data contains the .mp3s for (and .wav, .lat, and .sau) files needed for generating the final text output.  The processed text files from this data are stored in the respective place below.  

QANTA contains the processed text data for Quizbowl.

SearchQA contains the processed text data for Jeopardy.

Auto generated data is not stored in the interest of space (~300 GB). A seperate repository containing the code needed to generate audio data with Google Text to Speech and decode the data with Kaldi is provided at:
https://github.com/DenisPeskov/QBASR_GenerateData

###  Example of Quizbowl data (the same sentence across different speakers):
Speaker 1:
http://qbasr.umiacs.io/HumanOriginalData/HumanData_MultiSpeaker/0/2_0.mp3

Speaker 2:
http://qbasr.umiacs.io/HumanOriginalData/HumanData_MultiSpeaker/1/2_0.mp3

Auto Generated:
http://qbasr.umiacs.io/HumanOriginalData/HumanData_MultiSpeaker/999/-997_0.mp3

### Example of Jeopardy data (Alex Trebek's voice):
http://qbasr.umiacs.io/HumanOriginalData/HumanData_9.26.18_FullJeopardyGame/2_1.mp3

Jeopardy Show #7828 (http://www.j-archive.com/showgame.php?game_id=6112) is hand-parsed.  

Data is stored as: columnNumber_QuestionNumber  (with 1 not 0 index).  So 2_1 corresponds to the first question second column (Name the Novel): "If the picture was to alter, it was to alter. That was all...not one blossom of his loveliness would ever fade".
