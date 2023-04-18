# SST Attributions

This folder contains attributions for our SST models.
Most folders contain four subfolders - one for each model type. Each of these subfolders then contains directories with
attributions. In this case, the model-type subfolders are <code>base</code>, <code>medium</code>, <code>small</code>, and
<code>mini</code>.

* <code>additional</code> - attributions for a bert-base-cased model trained for 1, 2, 4, and 5 epochs
* <code>ig-test</code> - attributions for our Integrated Gradients baseline tests
* <code>sg-test</code> - attributions for our SmoothGRAD noise tests
* <code>ks-test</code> - attributions for our KernelSHAP baseline tests
* <code>final</code> - the final attributions we generated after determining the hyperparameters