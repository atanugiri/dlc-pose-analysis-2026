bash
```
rsync -avzP --exclude='.gitignore' --exclude='.git' \
--exclude='__pycache__' --exclude='.DS_Store' \
--dry-run . punakha:/work/agiri/dlc-pose-analysis-2026/
```

```
rsync -avzP --exclude='.gitignore' --exclude='.git' \
--exclude='__pycache__' --exclude='.DS_Store' \
--dry-run punakha:/work/agiri/dlc-pose-analysis-2026/ .
```
