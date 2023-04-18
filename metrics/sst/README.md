# SST Metrics

This folder contains metrics for our SST models.
Most folders contain four subfolders - one for each model type. Each of these subfolders then contains directories with
metrics for individual models and a file with merged metrics (always <code>metrics.csv</code>). 
In this case, the model-type subfolders are <code>base (bert-base-cased)</code>, <code>medium (prajjwal1/bert-medium)</code>,
<code>small (prajjwal1/bert-small)</code>, and <code>mini (prajjwal1/bert-mini)</code>.

* <code>additional</code> - metrics for a bert-base-cased model trained for 1, 2, 4, and 5 epochs.
* <code>ig-tests</code> - metrics for our Integrated Gradients baseline tests
* <code>sg-tests</code> - metrics for our SmoothGRAD noise tests
* <code>ks-tests</code> - metrics for our KernelSHAP baseline tests
* <code>final</code> - the final metrics we calculated after determining the hyperparameters