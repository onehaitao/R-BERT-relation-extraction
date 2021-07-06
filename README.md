# R-BERT-relation-extraction
Implementation of [Enriching Pre-trained Language Model Entity Information for Relation Classification](https://dl.acm.org/doi/abs/10.1145/3357384.3358119).

## Environment Requirements
* python 3.6.9
* pytorch 1.5.1
* transformers 2.11.0
* tqdm 4.40.1

## Data
* [SemEval2010 Task8](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50) \[[paper](https://www.aclweb.org/anthology/S10-1006.pdf)\]
* [bert-base-uncased](https://huggingface.co/bert-base-uncased)

## Usage
1. Download the pre-trained BERT model and put it into the `resource` folder.
2. Run the following the commands to start the program.
```shell
python run.py
```
More details can be seen by `python run.py -h`.

3. You can use the official scorer to check the final predicted result.
```shell
perl semeval2010_task8_scorer-v1.2.pl proposed_answer.txt predicted_result.txt >> result.txt
```

## Result
The result of my version and that in paper are present as follows:
| paper | my version |
| :------: | :------: |
| 0.8925 | 0.8313 |

The training log can be seen in `train.log` and the official evaluation results is available in `result.txt`.

*Note*:
* Some settings may be different from those mentioned in the paper.
* No validation set used during training.


## Reference Link
* https://github.com/monologg/R-BERT