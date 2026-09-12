python run.py train-all
python run.py prepare-data
python run.py train-tokenizer
python run.py tokenize-data

python run.py train-base
python run.py train-base --epochs 10 --dry-run

python run.py train-domain
python run.py train-domain --lr 0.0001 --epochs 5

python run.py finetune
python run.py finetune --epochs 5 --learning_rate 0.00005

python run.py import-qa
python run.py import-qa --stats-only

python run.py crawl-website
python run.py crawl-website --dry-run
python run.py refresh-website

python -m scripts.evaluate_model
python -m scripts.evaluate_finetuned_model
python -m scripts.evaluate_domain
python -m scripts.evaluate_website_knowledge
python -m scripts.evaluate_safety
python -m scripts.evaluate_e2e

python -m scripts.inspect_dataset
python -m scripts.inspect_domain_dataset
python -m scripts.inspect_tokenizer
python -m scripts.inspect_model
python -m scripts.inspect_conversations