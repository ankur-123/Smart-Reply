# Smart-Reply

1. Get Ubuntu corpus dataset from "https://s3.amazonaws.com/ngv-public/data.zip"

2. Install python 3.6 and install requirements from requirements.txt

pip install -r requirements.txt
source activate ml


3. Update the path variables with links to the data and where you want to save model output in main.py

parser.add_argument('--root_dir', default='')
parser.add_argument('--dataset_train_path', default='')
parser.add_argument('--dataset_test_path', default='')
parser.add_argument('--dataset_val_path', default='')
parser.add_argument('--vocab_path', default='')
parser.add_argument('--model_save_dir', default='')
parser.add_argument('--test_tube_dir', default='')

4. Run the code for training and testing
python main.py