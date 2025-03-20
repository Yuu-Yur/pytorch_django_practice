from django.core.files.uploadedfile import InMemoryUploadedFile
from django.shortcuts import render

# Create your views here.
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework import status, serializers, viewsets
from rest_framework.response import Response

from Pytorch_Django_Practice.forms import ImageUploadForm
from .models import load_model
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
# CNN 모델 로드
model = load_model()

# 클래스 이름
class_names = ["Hammer", "Nipper"]

# 이미지 전처리
def transform_image(image):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0) # 배치 차원 추가

# API: 이미지 업로드 및 분류
@csrf_exempt # csrf 예외 설정
def classify_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES["image"]
            image = Image.open(image_file)
            image = transform_image(image)

# 모델 예측
            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()

                response_data = {
                "class": class_names[predicted_idx],
                "confidence": round(confidence * 100, 2),
                }
                return JsonResponse(response_data)

    return JsonResponse({"error": "Invalid request"}, status=400)

def index(request):
    return render(request, "index.html")

# REST API 형식, 스프링, RESTController 형식이다.
# 이미지 변환 함수
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    return image
#임시로 , 시리얼라이저 클래스 등록하기.
# Serializer 정의
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()

# ViewSet 정의
class ImageClassificationViewSet(viewsets.ViewSet):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]
    #스웨거에서, 파일 업로드 안보일 때 설정 코드.

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                "image",
                openapi.IN_FORM,
                description="업로드할 이미지 파일",
                type=openapi.TYPE_FILE
            )
        ]
    )

    def create(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image_file = serializer.validated_data['image']
        if not isinstance(image_file, InMemoryUploadedFile):
            return Response({'error': 'Invalid file type'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image = Image.open(image_file).convert('RGB')
            image = transform_image(image)

            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()

            response_data = {
                'class': class_names[predicted_idx],
                'confidence': round(confidence * 100, 2)
            }
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
