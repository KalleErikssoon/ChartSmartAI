import os
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
import pandas as pd
import json
from stock_app.models import MACD_Data 
from stock_app.models import EMA_Data
from stock_app.models import RSI_Data
import os
from django.conf import settings

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
@parser_classes([MultiPartParser])
def upload_model(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        pkl_file = request.FILES['file']
        file_name = pkl_file.name

        # path to save the file
        save_path = os.path.join(settings.BASE_DIR, 'models', file_name)
        # make the directory if doesnt exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #save
        with open(save_path, 'wb') as f:
            for chunk in pkl_file.chunks():
                f.write(chunk)

        return JsonResponse({'message': 'File successfully uploaded'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
