# Prepare the environment
`pip install -r requirements.txt`

# prepare dataset
Our data processing is located in `./dataset/{dataset}.ipynb`. Simply modify the path and run it.

# train FreLLM4Rec

train id model:

`cd ./Pre_Train_Rec_Model/sasrec`

`python main.py --device=cuda --dataset {dataset}`

run FreLLM4Rec:

`sh run_main.sh`
