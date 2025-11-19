# SwinTExCo Module

此目錄應包含 SwinTExCo 變體的代碼。

## 如何獲取

從 swinthxco_single 倉庫複製以下文件到此目錄：

```bash
# 從 swinthxco_single 倉庫複製
cp -r swinthxco_single/src ./SwinSingle/
cp swinthxco_single/inference.py ./SwinSingle/
cp swinthxco_single/train.py ./SwinSingle/
```

## 必要修改

請確保已修改 SwinTExCo 以返回 `similarity_map`：

### 修改 1: src/models/CNN/FrameColor.py
```python
# 修改返回值從:
return IA_ab_predict, nonlocal_BA_lab

# 改為:
return IA_ab_predict, nonlocal_BA_lab, similarity_map
```

### 修改 2: train.py 和 inference.py
相應更新接收返回值。

## 目錄結構

```
SwinSingle/
├── src/
│   ├── models/
│   │   ├── CNN/
│   │   │   ├── FrameColor.py
│   │   │   ├── WarpNet.py
│   │   │   └── ...
│   │   └── Transformer/
│   └── utils/
├── inference.py
└── train.py
```
