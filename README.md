# GEMCNN

<img width="1799" height="941" alt="image" src="https://github.com/user-attachments/assets/d21a3f12-b377-4852-9b02-c0d297f0fdd7" />
GEMCNN can process measurements from four distinct data sources: FTIR and UV
(spectroscopy analysis) and ICP-MS and XRF (elemental analysis). b. With a negligible inference time
compared to human experts, it can predict the probability of a gemstone’s origin or whether it has undergone
heat treatment. Not all data types are required for inference; missing sources can be masked out as illustrated
by switch symbols in the figure. c. If the maximum probability exceeds a predefined threshold (top panel),
the stone prediction can be confidently accepted. If the maximum probability falls below the threshold
(bottom panel), however, the output should be discarded and the stone should be further analyzed via
standard methods such as microscopy and expert analysis. The value of the threshold, selected during the
confidence-thresholding phase, determines the balance between the number of stones that can be processed
automatically and the accuracy achieved by the model.

<img width="1543" height="580" alt="image" src="https://github.com/user-attachments/assets/2a6124ad-5973-4732-a388-f8dae72b7abe" />
Accuracy [%] vs. stones above the threshold [%] for TD with XRF (Left) and ICP
(right). Both XRF and ICP perform sub-optimally compared to other data sources. Data sources that are not
present in the legend are masked.

<img width="819" height="647" alt="image" src="https://github.com/user-attachments/assets/d4eb2928-96de-4abf-9b3d-9b1a9904e51d" />
Comparison between human experts (represented by crosses) and Gemtelligence (represented
by circles) in terms of the size of the subset of stones that have been confidently classified (on the x-axis)
and the corresponding level of accuracy achieved for this subset (on the y-axis). Each color corresponds to a
different combination of data sources. All the combinations apart from the red one (UV+FTIR) are used for
OD while UV+FTIR is used for TD. The dashed lines are used to highlight the performance change between
humans and our model. The results in the plot are obtained by evaluating the performance of experts and
Gemtelligence on test data.

<img width="1530" height="523" alt="image" src="https://github.com/user-attachments/assets/8a2d76ac-0496-42ee-b853-cfa6c8ee5c86" />
Confusion matrices for TD in the three considered operating modes, namely (Left)
None, (Middle) Mode 1, and (Right) Mode 2.

### Usage

```python
from gemcnn import GEMCNN, fit

model = GEMCNN(
    uv_input_len=1200, ftir_input_len=1700,
    xrf_features=30,   icpms_features=40,
    task='od',
)

# fit(model, train_loader, val_loader, device, uv_mean, ftir_mean, elem_mean)
```
