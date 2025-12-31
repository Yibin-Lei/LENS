import argparse
import logging
import os
import json
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

from mteb import MTEB
from model_wrapper import LENSWrapper
from instruction import task_to_instruction


TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "NFCorpus",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
    "ArguAna",
    "FiQA2018",
    "SCIDOCS",
    "QuoraRetrieval",
    "DBPedia",
    "ClimateFEVER",
    "FEVER",
    "HotpotQA",
    "MSMARCO",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "NQ",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
]

TASK_LIST_SUMMARIZATION = [
    "SummEval"
]

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Path to the trained model")
    parser.add_argument("--pooling_method", type=str, default='max-pooling')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bidirectional", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    return args

def save_results(task_results, task_name, results_dir):
    """Save results for a specific task list and calculate average."""
    # Calculate the average score for the task list
    average_score = sum(task_results.values()) / len(task_results) if task_results else 0
    
    # Prepare the output dictionary
    output_dict = {
        "tasks": task_results,
        "average": average_score
    }
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Define the output file path
    output_file = os.path.join(results_dir, f"{task_name}_results.json")
    
    # Write the results to the JSON file
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    # Log the results
    logging.info(f"Results for {task_name}:")
    logging.info(f"Average score: {average_score:.3f}")

def main(args):
    model = LENSWrapper(model_name_or_path=args.model_name_or_path, batch_size=args.batch_size,
                        pooling_method=args.pooling_method, bidirectional=args.bidirectional)

    # Dictionary to store results for each task
    task_results = defaultdict(dict)
    task_lists = [
        ("STS", TASK_LIST_STS),
        ("Summarization", TASK_LIST_SUMMARIZATION),
        ("PairClassification", TASK_LIST_PAIR_CLASSIFICATION),
        ("Reranking", TASK_LIST_RERANKING),
        ("Retrieval", TASK_LIST_RETRIEVAL),
        ("Clustering", TASK_LIST_CLUSTERING),
        ("Classification", TASK_LIST_CLASSIFICATION),
    ]

    model_identifier = f"{args.model_name_or_path.split('/')[-1]}"
    results_dir = f"./mteb_results/{model_identifier}"
    print(f"Results dir: {results_dir}")
    
    for task_name, task_list in task_lists:
        print(f"\nEvaluating {task_name} tasks:")
        for task in task_list:
            print(f"Running task: {task}")
            eval_splits = ["dev"] if task == "MSMARCO" else ["test"]

            evaluation = MTEB(tasks=[task], task_langs=[args.lang])
            
            results = evaluation.run(
                model,
                output_folder=f"{results_dir}/{task_name}",
                batch_size=args.batch_size,
                eval_splits=eval_splits,
                overwrite_results=True,
            )
            
            # Store the main metric for each task
            main_metric = results[0].scores['test'][0]['main_score']
            task_results[task_name][task] = main_metric
        
        # Save results for each task list
        save_results(task_results[task_name], task_name, results_dir)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)