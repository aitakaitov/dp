# CTDC Metrics

This folder contains metrics for our CTDC models.
Most folders contain four subfolders - one for each model type. Each of these subfolders then contains directories with
metrics for individual models and a file with merged metrics (always <code>metrics.csv</code>). In this case, the model-type subfolders are <code>czert (UWB-AIR/Czert-B-base-cased)</code>, 
<code>minilm (nreimers/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large)</code>, and <code>seznam (Seznam/small-e-czech)</code>.

* <code>ks-size</code> - KernelSHAP metrics for a small-e-czech model with n=1000 and n=2000. 
* <code>ig-tests</code> - metrics for our Integrated Gradients baseline tests
* <code>sg-tests</code> - metrics for our SmoothGRAD noise tests
* <code>ks-tests/code> - metrics for our KernelSHAP baseline tests
* <code>final</code> - the metrics we generated after determining the hyperparameters