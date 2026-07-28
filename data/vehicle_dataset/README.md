# Vehicle Dataset

Put your labeled YOLO dataset here when you are ready to train a custom vehicle model.

Expected layout:

```text
data/vehicle_dataset/
|-- dataset.yaml
|-- images/
|   |-- train/
|   `-- val/
`-- labels/
    |-- train/
    `-- val/
```

Class IDs in label `.txt` files should match `dataset.yaml`:

```text
0 car
1 motorcycle
2 bus
3 truck
```

Train with:

```powershell
python tools\train_vehicle_model.py --data data\vehicle_dataset\dataset.yaml
```
