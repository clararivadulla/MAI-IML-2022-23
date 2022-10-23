# IML-Work1

Work 1 for Intro to Machine Learning course (Masters for Artificial Intelligence)

Steps to run the script:


### TEAM 1 

Shawn Azimov, Jana Dabovic, Clara Rivadulla, and Yara Yousef


### Runnning script for the first time
### These sections show how to create virtual environment for our script and how to install dependencies
1. Open folder in terminal
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

### How to run the main file - main file runs all 3 datasets (Iris, Cmc, and Pima diabetes) on 6 algorithms (Agglomerative clustering, Mean shift, KMeans, K-Bisecting means, K-harmonic means, Fuzzy C-Means) using parameters that we identified as optimal for each of the algorithms
```bash
python main.py
```
### How to run algorithms on particular dataset - each dataset has its own main file that runs above mentioned 6 algorithms with optimized parameters - here you can change parameters of each algorithm per dataset
#### Run iris main file
```bash
python main_iris.py
```
#### Run cmc main file
```bash
python main_cmc.py
```
#### Run pima diabetes main file
```bash
python main_pima_diabetes.py
```
### How to run the performance scripts - this script runs all algorithms on a list of possible k values (number of clusters) on a specified dataset (dataset is specified at the botom of the script) - the output plots performance of algorithms using 4 scores (Silhouette, Davies Bouldin, Calinski Harabasz and Adjusted Mutual Info score) against different values of k
1. Go into performance directory
```bash
cd performance
```
2. Run test_performance.py 
```bash
python test_performance.py
```

