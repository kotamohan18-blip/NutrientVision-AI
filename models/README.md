# Required trained checkpoint

Place the trained project checkpoint here:

`models/densenet121_best.pth`

The checkpoint is produced by notebook Cell 118 and saved with:

```python
torch.save(densenet_model.state_dict(), best_model_path)
```

Expected architecture:
- torchvision DenseNet121
- classifier: `nn.Linear(1024, 6)`
- class order: B12, Iron, Vitamin_A, Vitamin_C, Vitamin_D, Zinc

Do not place a generic ImageNet checkpoint here. It will not contain the trained six-class project classifier.
