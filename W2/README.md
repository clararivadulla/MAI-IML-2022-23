# IML-Work2

## Work 2 for Intro to Machine Learning course (Masters for Artificial Intelligence)

### TEAM 1
Shawn Azimov, Jana Dabovic, Clara Rivadulla, and Yara Yousef


These sections show how to create virtual environment, install dependencies and run the script.
The script will preprocess the datasets, and for each dataset it will use various feature reduction algorithms.
For each of the algorithms, it will run k-means and agglomerative clustering on transformed datasets. 
It will also generate 2-D and 3-D (where appropriate) scatter plots of the data at <root_folder_of_project>/figures/scatter-plots

1. Navigate to the folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env using python 3.7
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```

5. Run the main file 
```bash
python main.py
```
