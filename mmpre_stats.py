import glob
import os.path as osp
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional

import pandas as pd
import pdfplumber
import tqdm
from pdfminer.pdfparser import PDFSyntaxError
from mmengine.utils import track_parallel_progress


@dataclass
class paper_info:
    title: str
    conference: str
    year: int
    authors: str
    abstract: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_url: Optional[str] = None
    code_url: Optional[str] = None


def load_paper_info(index_path: str = 'index'):
    """Load paper information from csv files which can be downloaded from
    https://aicarrier.feishu.cn/sheets/shtcnGhSBiEUVqnHQBPtshy6Tse and should
    be organized as:

    index
    ├── 顶会论文数据库-AAAI.csv
    ├── 顶会论文数据库-CVPR.csv
    ├── 顶会论文数据库-ECCV.csv
    ...
    """
    papers = []

    for fn in glob.glob(osp.join(index_path, '*.csv')):
        print(f'load paper index from {fn}')
        df = pd.read_csv(fn)

        for _, item in df.iterrows():
            paper = paper_info(title=item['title'],
                               conference=item['conference'],
                               year=item['year'],
                               authors=item['authors'],
                               abstract=item['abstract'])

            if isinstance(item['pdf_url'], str):
                paper.pdf_url = item['pdf_url']

            if isinstance(item['code_url'], str):
                paper.code_url = item['code_url']

            paper.pdf_path = osp.join('data',
                                      f'{paper.conference}{paper.year}',
                                      f'{paper.title}.pdf')

            papers.append(paper)

    print(f'load {len(papers)} papers in total.')
    return papers


def _download(args):
    import wget
    idx, total, paper = args
    url = paper.pdf_url
    path = paper.pdf_path
    try:
        print(f'{idx}/{total}')
        wget.download(url, path, bar=None)
        return None
    except:  # noqa
        return paper


def download_missing_pdf(missing_list):

    with Pool() as p:
        total = len(missing_list)
        tasks = [(i, total, paper) for i, paper in enumerate(missing_list)]
        failed_list = [r for r in p.map(_download, tasks) if r is not None]

    if failed_list:
        print(f'failed to download {len(failed_list)} papers.')
        with open('failed_list.pkl', 'wb') as f:
            pickle.dump(failed_list, f)


def search_kwgroups_in_pdf(pdf_path: str,
                           keyword_groups: dict[str, list[str]],
                           case_sensitive=False) -> list[int]:
    """Search a keyword groups in a pdf file. One keyword group is considered
    hit if at least one keyword in this group is found in the pdf.

    Args:
        pdf_path (str): path to the pdf file
        keyword_groups (dict[str, list[str]]): A list of keyword groups. Each
            group is a list of keywords
        case_sensitive (bool): Whether consider letter case
    
    Returns:
        dict[str, bool]: The indicators of each keypoint group.
    """

    if not case_sensitive:
        keyword_groups = {
            k: [kw.lower() for kw in group]
            for k, group in keyword_groups.items()
        }

    result = {k: False for k in keyword_groups.keys()}
    if osp.isfile(pdf_path):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for _, page in enumerate(pdf.pages, 1):
                    if all(result.values()):
                        break

                    text = page.extract_text()
                    if not case_sensitive:
                        text = text.lower()

                    for name, group in keyword_groups.items():
                        if result[name]:
                            continue
                        else:
                            for kw in group:
                                if kw in text:
                                    result[name] = True
                                    break
        except PDFSyntaxError:
            print(f'fail to parse: {pdf_path}')

    return result


def _search_in_pdf(args):
    idx, total, keyword_groups, paper = args
    return paper, search_kwgroups_in_pdf(paper.pdf_path, keyword_groups)


def main():
    # load paper information
    papers = load_paper_info()
    missing_list = [
        paper for paper in papers if not osp.isfile(paper.pdf_path)
    ]
    print(f'found {len(missing_list)} missing papers.')
    # download_missing_pdf(papers)

    # search in title/abstract
    def _valid(paper):
        pos_kws = [
            'network',
            'backbone',
            'classification',
            'self-supervised',
            'selfsup',
            'retrieval',
            'vision transformer',
            'resnet',
        ]

        neg_kws = []

        text = paper.title.lower()
        # if isinstance(paper.abstract, str):
        #     text = text + ' ' + paper.abstract.lower()

        for kw in neg_kws:
            if kw in text:
                return False

        for kw in pos_kws:
            if kw in text:
                return True

        return False

    papers = list(filter(_valid, papers))
    print(f'Select {len(papers)} papers first.')

    # search in PDF
    keyword_groups = {
        '+backbone': ['vision backbone', 'vision network'],
        '+task': ['classification', 'self-supervised', 'selfsup', 'retrieval'],
        '+resnet': ['resnet', 'residual connenction'],
        '+vit': [
            'vit', 'vision transformer', 'swin-transformer',
            'swin transformer', 'vision-transformer', 'convnext'
        ],
        'mmpretrain': ['mmpretrain', 'mmcls', 'mmclassfication', 'mmselfsup'],
        'openmmlab': ['openmmlab', 'open mmlab', 'open-mmlab'],
        'timm': ['timm', 'pytorch-image-models', 'pytorch image models'],
        'paddle': ['paddleclas', 'paddle-clas'],
    }

    total = len(papers)
    tasks = [(i, total, keyword_groups, paper)
             for i, paper in enumerate(papers)]
    search_results = track_parallel_progress(_search_in_pdf, tasks, 32, keep_order=False)

    with open('mmpre_search_results.pkl', 'wb') as f:
        pickle.dump(search_results, f)

    matched = []
    for paper, result in search_results:
        pos_keys = [k for k in result.keys() if k.startswith('+')]
        neg_keys = [k for k in result.keys() if k.startswith('-')]

        relevant = False
        for key in pos_keys:
            relevant |= result.pop(key)

        for key in neg_keys:
            relevant &= (~result.pop(key))

        result['relevant'] = relevant
        result = {k: int(v) for k, v in result.items()}
        if any(result.values()):
            matched.append((paper, result))

    for name in matched[0][1].keys():
        count = sum(result[name] for _, result in matched)
        print(name, count)

    # save to csv
    paper_dicts = []
    for paper, result in matched:
        d = paper.__dict__.copy()
        d.update(result)
        paper_dicts.append(d)

    df = pd.DataFrame.from_dict(paper_dicts)
    df.to_csv('mmpre_stats.csv', index=False, header=True)


if __name__ == '__main__':

    main()
