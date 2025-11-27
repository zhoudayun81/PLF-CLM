# Process Log Forecasting using Causal Language Modelling
Our code uses Python3, to ensure smooth running, please make sure your python environment is at least Python 3.9.19.

We also used Optuna (https://optuna.org/). Other package information is available in our paper.

## Running experiment
A quick cheating command to install all packages using pip to run our code is: 
<pre>
  <code id="install-command">pip install torch optuna transformers datasets packaging nltk scikit-learn numpy</code>
</pre>
<button onclick="copyToClipboard('#install-command')"></button>
This should allow you to install all the required packages at once.

Please make sure you keep the folder structures under this main folder [code folder](https://github.com/zhoudayun81/PLF-CLM/tree/main/) (```.\```), run the following to start the experiment: 
<pre>
  <code id="install-command">python3 main.py</code>
</pre>
<button onclick="copyToClipboard('#install-command')"></button>
For baseline models, simply change the line in main ```evals = process_file(traces, filename, config, device)``` to ```evals = process_file_baseline(traces, filename, config, device)```:

You can modify the settings to different parameters for other experiment, and we left the [configuration file](https://github.com/zhoudayun81/PLF-CLM/tree/main/config.ini) (```.\config.ini```) as the default settings that we used and described in our paper. Explanations are provided in the file for each parameter that may not be intuitive to understand.

If you want to run cross-validation (multiple folds), you need to modify the split ratio ```"split_ratios": [0.8, 0.1, 0.1],``` in the config file to the desired portion, and also save the prediction outputs and evaluations to different folders to avoid previous results being rewritten.

## Accessing results
You can find our experiment prediction outputs in the [output folder](https://github.com/zhoudayun81/PLF-CLM/tree/main/pred) and we summarize the results in excels in the [result folder](https://github.com/zhoudayun81/PLF-CLM/tree/main/download/result).
For interpretations to the results, please refer to our paper.
<hr />

Please note that our licence is under GNU Affero General Public License v3.0. 
If you need assistance in running the code or you are interested in the experiment or further collaboration, feel free to contact [Wenjun](mailto:wenjun.zhou@unimelb.edu.au?subject=[PLF-CLM]).
