# Robust Subspace Structure Discovering for scRNA-seq Cell Type Identification
This is a Pytorch implementaion of Robust Subspace Structure Discovering for scRNA-seq Cell Type Identification, as described in our paper:
...

## Results

In this repo, we provide Robust Subspace Structure Discovering for scRNA-seq Cell Type Identification code with the human_ESC,human_brain and time_course datasets as example. 

| Dataset | ACC  | NMI | | ARI |
| human_ESC | 1.00 | 1.00 | 1.00 | 
| human_brain  | 0.8833 | 0.8568 | 0.8679 | 
| time_course| 0.9155 | 0.8908 | 0.8062 | 


## Usage
# Step 1:Prepare pytorch environment.
python 3.8.13
pytorch 1.10.2
scikit-learn 1.1.1
numpy 1.22.3
cvxpy 1.2.0
scipy 1.9.1

# Step 2:Prepare data, use your data or ours data(human_ESC,human_brain,time_course)

# Step 3:Train the local graph to guide the weights in Robust extract.py,If you use your own dataset, the weight size will affect the clustering metrics,Suggest starting from weight=100.
(warming:You need to activate it yourself, otherwise it won't run. The URL is this https://www.mosek.com/)

# Step 4:Run on human_ESC,human_brain,time_course by using the following commands.

```
python main.py --db human_ESC 
python main.py --db human_brain
python main.py --db time_course
```
(warming:Each parameter indicator in the code will have an impact on the clustering indicator. If you use your own data, you need to adjust the parameters to achieve the best clustering effect)


## Cite

Please cite our paper if you use this code in your own work:...






## Cite

Please cite our paper if you use this code in your own work:...

