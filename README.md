# Paper-stats

## Data Organization

Please download and organize data as the following:

- `data`: paper PDF files
- `index`: Raw paper list in csv files
- `stats`: Retrived paper information in csv files (generated by `mmpose_stats.py` and `mmaction2_stats.py`)
- `index_with_tags`: Paper list with area/codebase tags in csv files (generated by `add_tags.py`)

![img](./data_organization.png)

## Usage

### Search related papers by keywords

```bash
# search papers on pose estimation
python mmpose_stats.py
# search papers on video understanding
python mmaction2_stats.py
```

### Add Tags

```bash
# generate paper index with tags
python add_tags.py
```