from datetime import datetime, timedelta
import os
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser
import pandas as pd
import json
from stock_app.models import MACD_Data 
from stock_app.models import EMA_Data
from stock_app.models import RSI_Data
import os
from django.conf import settings
from stock_app.inference.inference import fetch_stock_data, load_model, run_inference, preprocess_data

from rest_framework.response import Response
from kubernetes import client, config
import subprocess
import time 
from kubernetes.client.rest import ApiException

import os

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_macd(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        # Read the uploaded file
        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file)

        # Insert each row into the database
        for _, row in df.iterrows():
            MACD_Data.objects.create(
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                trade_count=row['trade_count'],
                vwap=row['vwap'],
                macd=row['macd'],
                signal_line=row['signal_line'],
                label=row['label']
            )
        return JsonResponse({'message': 'Data successfully uploaded'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_ema(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        # Read the uploaded file
        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file)

        # Insert each row into the database
        for _, row in df.iterrows():
            EMA_Data.objects.create(
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                trade_count=row['trade_count'],
                vwap=row['vwap'],
                ema=row['ema'],
                label=row['label']
            )
        return JsonResponse({'message': 'Data successfully uploaded'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_rsi(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        # Read the uploaded file
        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file)

        # Insert each row into the database
        for _, row in df.iterrows():
            RSI_Data.objects.create(
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                trade_count=row['trade_count'],
                vwap=row['vwap'],
                rsi=row['rsi'],
                label=row['label']
            )
        return JsonResponse({'message': 'Data successfully uploaded'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# GET endpoint to get the processed data for the ML model for EMA
@api_view(['GET'])
def get_ema_data(request):
    try:
        # Get data from SQLite DB
        macd_data = EMA_Data.objects.all()
        # Convert results above to a list of dictionaries
        data = list(macd_data.values(
            'symbol', 'timestamp', 'open', 'high', 'low', 
            'close', 'volume', 'vwap', 'trade_count',
            'ema', 'label'
        ))

        df = pd.DataFrame(data)

        # create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="ema_data.csv"'

        # Write dataframe to the response as CSV
        df.to_csv(path_or_buf=response, index=False)

        return response

    except Exception as e:
        # handle exceptions and return an error response
        return HttpResponse(f"Error: {str(e)}", content_type="text/plain", status=500)


# GET endpoint to get the processed data for the ML model for MACD
@api_view(['GET'])
def get_macd_data(request):
    try:
        # Get data from SQLite DB
        macd_data = MACD_Data.objects.all()
        # Convert results above to a list of dictionaries
        data = list(macd_data.values(
            'symbol', 'timestamp', 'open', 'high', 'low', 
            'close', 'volume', 'trade_count', 'vwap', 
            'macd', 'signal_line', 'label'
        ))

        df = pd.DataFrame(data)

        # create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="macd_data.csv"'

        # Write dataframe to the response as CSV
        df.to_csv(path_or_buf=response, index=False)

        return response

    except Exception as e:
        # handle exceptions and return an error response
        return HttpResponse(f"Error: {str(e)}", content_type="text/plain", status=500)
    


    # GET endpoint to get the processed data for the ML model for RSI
@api_view(['GET'])
def get_rsi_data(request):
    try:
        # Get data from SQLite DB
        rsi_Data = RSI_Data.objects.all()
        # Convert results above to a list of dictionaries
        data = list(rsi_Data.values(
            'timestamp', 'symbol', 'open', 'high', 'low', 
            'close', 'volume', 'vwap', 'trade_count',
            'rsi', 'label'
        ))

        df = pd.DataFrame(data)

        # create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="rsi_data.csv"'

        # Write dataframe to the response as CSV
        df.to_csv(path_or_buf=response, index=False)

        return response

    except Exception as e:
        # handle exceptions and return an error response
        return HttpResponse(f"Error: {str(e)}", content_type="text/plain", status=500)

@csrf_exempt
@api_view(['POST'])
def upload_metadata(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    uploaded_file = request.FILES['file']

    try:
        # Read the file content and parse it as JSON
        metadata = json.load(uploaded_file)

        # Get the original file name
        original_filename = uploaded_file.name

        # Ensure the metadata directory exists
        metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        # Construct the full path using the original file name
        metadata_file_path = os.path.join(metadata_dir, original_filename)

        # Save the metadata to the specified file
        with open(metadata_file_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

        return JsonResponse({'message': 'File uploaded successfully', 'metadata': metadata}, status=201)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON file'}, status=400)

@csrf_exempt
@api_view(['POST'])
def rename_metadata(request):
    # rename the metadata file to the new name
    data = request.data
    old_name = data.get('old_name')
    new_name = data.get('new_name')

    # Rename the metadata file in the metadata directory
    metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../metadata")
    old_path = os.path.join(metadata_dir, old_name)
    new_path = os.path.join(metadata_dir, new_name)

    try:
        os.rename(old_path, new_path)
        return JsonResponse({'message': 'File renamed successfully'}, status=200)
    except FileNotFoundError:
        return JsonResponse({'error': 'File not found'}, status=404)
    except FileExistsError:
        return JsonResponse({'error': 'File with the new name already exists'}, status=409)


# @csrf_exempt
# @api_view(['POST'])
# @parser_classes([MultiPartParser])
# def upload_model(request):
#     if 'file' not in request.FILES:
#         return JsonResponse({'error': 'No file uploaded'}, status=400)

#     try:
#         pkl_file = request.FILES['file']
#         file_name = pkl_file.name

#         # path to save the file
#         save_path = os.path.join(settings.BASE_DIR, 'models', file_name)
#         # make the directory if doesnt exist
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         #save
#         with open(save_path, 'wb') as f:
#             for chunk in pkl_file.chunks():
#                 f.write(chunk)

#         return JsonResponse({'message': 'File successfully uploaded'}, status=201)

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_model(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        pkl_file = request.FILES['file']
        file_name = pkl_file.name

        strategy = "-"

        if "macd" in file_name:
            strategy = "macd"
        elif "ema" in file_name:
            strategy = "ema"
        elif "rsi" in file_name:
            strategy = "rsi"

        # path to save the file
        save_path = os.path.join(settings.BASE_DIR, f"stock_app/inference/models/{strategy}/", file_name)
        # make the directory if doesnt exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #save
        with open(save_path, 'wb') as f:
            for chunk in pkl_file.chunks():
                f.write(chunk)

        return JsonResponse({'message': 'File successfully uploaded'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser, MultiPartParser])
def retrain(request):
    data = request.data
    strategy = data.get('strategy')
    if not strategy:
        return Response({'error': 'No strategy provided'}, status=400)

    if isinstance(strategy, list):
        strategy = strategy[0]

    config.load_kube_config()

    strategy_lower = strategy.lower()
    if strategy_lower == "rsi":
        run_rsi_strategy_job()
    elif strategy_lower == "macd":
        run_macd_strategy_job()
    elif strategy_lower == "ema":
        run_ema_strategy_job()
    else:
        return Response({'error': f'Unknown strategy: {strategy}'}, status=400)
    # time.sleep(3)
    #after running strategy pipeline job run the model job
    run_model_job(strategy)
    
    return Response({'message': f'Retraining job for {strategy} created successfully'}, status=200)

def run_rsi_strategy_job():
    # RSI job definition
    subprocess.run(["python", "./stock_project/scripts/clear_rsi_table.py"], check=True)
    clear_job("rsi-pipeline-job", namespace="default")
    job_spec = {
        "api_version": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "rsi-pipeline-job"
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "rsi-pipeline",
                        "image": "gcr.io/adroit-arcana-443708-m9/rsi_pipeline:v1",
                        "imagePullPolicy": "Always",
                        "command": ["python", "main_script.py"] 
                    }],
                    "restartPolicy": "Never"
                }
            }
        }
    }

    batch_v1 = client.BatchV1Api()
    job = client.V1Job(**job_spec)
    batch_v1.create_namespaced_job(namespace="default", body=job)

def run_macd_strategy_job():
    # MACD job definition
    subprocess.run(["python", "./stock_project/scripts/clear_macd_table.py"], check=True)
    clear_job("macd-pipeline-job", namespace="default")
    job_spec = {
        "api_version": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "macd-pipeline-job"
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "macd-pipeline",
                        "image": "gcr.io/adroit-arcana-443708-m9/macd_test:v1",
                        "command": ["python", "main_script.py"]
                    }],
                    "restartPolicy": "Never"
                }
            }
        }
    }

    batch_v1 = client.BatchV1Api()
    job = client.V1Job(**job_spec)
    batch_v1.create_namespaced_job(namespace="default", body=job)


def run_ema_strategy_job():
    # ema  job definition
    subprocess.run(["python", "./stock_project/scripts/clear_ema_table.py"], check=True)
    clear_job("ema-pipeline-job", namespace="default")
    job_spec = {
        "api_version": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "ema-pipeline-job"
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "ema-pipeline",
                        "image": "gcr.io/adroit-arcana-443708-m9/ema_pipeline:v1",
                        "imagePullPolicy": "Always",
                        "command": ["python", "main_script.py"]
                    }],
                    "restartPolicy": "Never"
                }
            }
        }
    }

    batch_v1 = client.BatchV1Api()
    job = client.V1Job(**job_spec)
    batch_v1.create_namespaced_job(namespace="default", body=job)
 

def run_model_job(strategy):
    # model job definition
    clear_job("model-job", namespace="default")
    job_spec = {
        "api_version": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "model-job"
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "model-implementation",
                        "image": "gcr.io/adroit-arcana-443708-m9/model_implementation:v1",
                        "imagePullPolicy": "Always",
                        "command": ["python", "main_script.py", strategy.lower()]
                    }],
                    "restartPolicy": "Never"
                }
            }
        }
    }

    batch_v1 = client.BatchV1Api()
    job = client.V1Job(**job_spec)
    batch_v1.create_namespaced_job(namespace="default", body=job)
    

    
def clear_job(job_name, namespace="default"):
    try:
        config.load_kube_config()
        batch_v1 = client.BatchV1Api()
        core_v1 = client.CoreV1Api()

        #check if job exists
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                print(f"Job '{job_name}' not found in namespace '{namespace}'. Skipping deletion.")
                return
            else:
                raise

        #delete the job and its pods
        delete_options = client.V1DeleteOptions(propagation_policy="Foreground")
        batch_v1.delete_namespaced_job(name=job_name, namespace=namespace, body=delete_options)
        print(f"Initiated deletion of job '{job_name}' in namespace '{namespace}'.")

        #wait for the job to be fully deleted
        while True:
            try:
                batch_v1.read_namespaced_job(name=job_name, namespace=namespace)
                print(f"Waiting for job '{job_name}' to be deleted...")
                time.sleep(3)
            except ApiException as e:
                if e.status == 404:
                    print(f"Job '{job_name}' successfully deleted.")
                    break
                else:
                    raise

        # make sure pods are deleted
        pod_list = core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_name}")
        for pod in pod_list.items:
            core_v1.delete_namespaced_pod(name=pod.metadata.name, namespace=namespace)
            print(f"Deleted pod '{pod.metadata.name}' associated with job '{job_name}'.")

    except ApiException as e:
        print(f"An error occurred: {e}")
        raise

@api_view(['GET'])
def list_files(request):
    import os
    directory_path = "./stock_project/stock_app/inference/models/"
    try:
        all_files = []  #list to collect all files across strategies
        strategies = ['rsi', 'ema', 'macd']

        for strategy in strategies:
            strategy_path = os.path.join(directory_path, strategy)
            
            # Skip if dir doesnt exist
            if not os.path.exists(strategy_path) or not os.path.isdir(strategy_path):
                print(f"Skipping non-existent folder: {strategy_path}")
                continue
            
            # exclude '_scaler.pkl'
            strategy_files = [
                f for f in os.listdir(strategy_path)
                if os.path.isfile(os.path.join(strategy_path, f)) and not f.endswith('_scaler.pkl')
            ]
            
            print(f"Checking path: {strategy_path}")
            all_files.extend(strategy_files)  

        return JsonResponse({"files": all_files}, status=200)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['POST'])
def change_chosen_model(request):
    try:
        #get chosen_strategy and chosen_model from the request data
        chosen_strategy = request.data.get('chosen_strategy')
        chosen_model = request.data.get('chosen_model')

        if not chosen_strategy or not chosen_model:
            return Response({'error': 'Both text and name are required.'}, status=400)

        #determine the file path based on the name
        if 'ema' in chosen_model.lower():
            file_path = "./stock_project/chosen_model/ema.txt"
        elif 'rsi' in chosen_model.lower():
            file_path = "./stock_project/chosen_model/rsi.txt"
        elif 'macd' in chosen_model.lower():
            file_path = "./stock_project/chosen_model/macd.txt"
        else:
            return Response({'error': 'Invalid name provided.'}, status=400)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        #save
        with open(file_path, 'w') as file:
            file.write(chosen_model + '\n')

        return Response({'message': 'Text saved successfully.'}, status=200)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
    

@api_view(['GET'])
def get_performance(request):
    directory_path = "./stock_project/metadata"
    print(directory_path)
    try:
        # List JSON files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.json') and not f.startswith('data_')]
        
        performance_data = []
        for file in files:
            file_path = os.path.join(directory_path, file)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                model_pickle = data.get("model_pickle")
                performance_metrics = data.get("performance metrics", {})
                classification_report = performance_metrics.get("classification_report", {})
                accuracy = classification_report.get("accuracy")
                macro_avg = classification_report.get("macro avg")
                performance_data.append({
                    "model_pickle": model_pickle,
                    "file": file,
                    "accuracy": accuracy,
                    "macro_avg": macro_avg
                })
        
        return JsonResponse({"performance_data": performance_data})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
@api_view(['GET'])
def make_prediction(request, strategy, stock_symbol):
    try:
        strategy = strategy.lower()
        print(f"make_prediction called with strategy={strategy} and stock_symbol={stock_symbol}")
        
        # Validate inputs
        if not stock_symbol or not strategy:
            return JsonResponse({"error": "Missing stock_symbol or strategy."}, status=400)
        
        # Define the mapping of user-friendly stock names to Alpaca-compatible symbols
        stock_symbol_mapping = {
            "Nvidia": "NVDA",
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Amazon": "AMZN",
            "Google": "GOOGL",
            "Meta": "META",
            "Tesla": "TSLA",
            "Berkshire Hathaway": "BRK.B",
            "Taiwan Semiconductors": "TSM",
            "Broadcom": "AVGO"
        }

        # Map the user-selected stock name to the Alpaca-compatible symbol
        alpaca_symbol = stock_symbol_mapping.get(stock_symbol)
        if not alpaca_symbol:
            return JsonResponse({"error": f"Invalid stock symbol: {stock_symbol}"}, status=400)

        # Fetch stock data
        print("Fetching stock data...")
        yesterday = (datetime.now() - timedelta(days=1)).date()
        start_date = yesterday - timedelta(days=15)  # Fetch at least 15 days of data for EMA
        stock_data = fetch_stock_data(alpaca_symbol, start_date=start_date, end_date=yesterday)
        
        if stock_data.empty:
            raise ValueError(f"No data fetched for {alpaca_symbol} from {start_date} to {yesterday}.")
        
        print(f"Fetched stock data:\n{stock_data.head()}")

        # Preprocess the stock data
        processed_data = preprocess_data(stock_data, strategy)
        print(f"Processed data:\n{processed_data}")

        # Run inference
        predicted_action = run_inference(processed_data, strategy)
        print(f"Predicted action: {predicted_action}")

        # Return the predictions as a response
        return JsonResponse({
            "stock_symbol": stock_symbol,
            "strategy": strategy,
            "prediction": predicted_action
        })

    except Exception as e:
        print(f"Error in make_prediction: {e}")
        return JsonResponse({"error": str(e)}, status=500)

