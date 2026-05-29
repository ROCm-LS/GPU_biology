# GPU_biology
AMD MI250x ready container recipes and test cases for biology tools

## Using the vram monitoring
1. submit the job to slurm, see which node it's assigned to
2. login to the node
3. run with bash vram_monitoring.sh job_id

You can change the print frequency by changing the sleep interval in the script.
